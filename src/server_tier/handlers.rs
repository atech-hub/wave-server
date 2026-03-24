//! Request handler functions — connects api_types, prompt, and inference.

use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::State;
use axum::response::{IntoResponse, Json, Response};
use axum::response::sse::{Event, Sse};
use axum::http::StatusCode;
use tokio_stream::wrappers::ReceiverStream;

use crate::api_types::*;
use crate::inference::{self, GenerationConfig, MemoryOffsets};
use crate::prompt;
use crate::server::AppState;

/// Log request telemetry to requests.jsonl (survives crashes, enables replay)
fn log_request(prompt_tokens: usize, gen_tokens: usize, time_ms: u128,
               temperature: f32, top_p: f32, text: &str) {
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("requests.jsonl") {
        let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
            .unwrap().as_secs();
        let tps = if time_ms > 0 { gen_tokens as f64 / time_ms as f64 * 1000.0 } else { 0.0 };
        let preview: String = text.chars().take(100).collect();
        let preview = preview.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n");
        let _ = writeln!(f, "{{\"ts\":{ts},\"prompt_tok\":{prompt_tokens},\"gen_tok\":{gen_tokens},\"ms\":{time_ms},\"tok_s\":{tps:.1},\"temp\":{temperature},\"top_p\":{top_p},\"text\":\"{preview}\"}}");
    }
}

/// POST /v1/chat/completions
pub async fn handle_chat_completion(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, StatusCode> {
    if req.messages.is_empty() {
        let err = ErrorResponse {
            error: ErrorDetail {
                message: "messages array must not be empty".to_string(),
                r#type: "invalid_request_error".to_string(),
            },
        };
        return Ok((StatusCode::BAD_REQUEST, Json(err)).into_response());
    }

    let prompt_tokens = prompt::format_chat(&req.messages, &state.vocab);
    if prompt_tokens.is_empty() {
        let err = ErrorResponse {
            error: ErrorDetail {
                message: "prompt encoded to zero tokens".to_string(),
                r#type: "invalid_request_error".to_string(),
            },
        };
        return Ok((StatusCode::BAD_REQUEST, Json(err)).into_response());
    }

    // Truncate prompt to block_size if needed
    let block_size = state.model.config.block_size;
    let prompt_tokens = if prompt_tokens.len() > block_size {
        prompt_tokens[prompt_tokens.len() - block_size..].to_vec()
    } else {
        prompt_tokens
    };

    let config = GenerationConfig {
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        repetition_penalty: req.repetition_penalty,
    };

    let model_name = state.config.model_name.clone();
    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let temp = req.temperature;
    let top_p = req.top_p;
    let prompt_len = prompt_tokens.len();

    if req.stream {
        Ok(handle_streaming(state, prompt_tokens, config, model_name, request_id, created, prompt_len, temp, top_p).await)
    } else {
        Ok(handle_non_streaming(state, prompt_tokens, config, model_name, request_id, created, prompt_len, temp, top_p).await)
    }
}

/// Build MemoryOffsets from AppState (if memory is active).
fn get_memory_offsets(state: &AppState) -> Option<MemoryOffsets> {
    let guard = state.memory.lock().unwrap();
    guard.as_ref().map(|mem| {
        let alpha = mem.config.alpha;
        MemoryOffsets {
            offsets: mem.layers.iter()
                .map(|l| l.scaled_offsets(alpha))
                .collect(),
        }
    })
}

/// Update persistent memory after a conversation.
/// Extracts ODE states from the generated sequence and merges via EMA.
fn update_memory(state: &AppState, all_tokens: &[usize], mem_offsets: Option<&MemoryOffsets>) {
    let mut guard = state.memory.lock().unwrap();
    let Some(ref mut memory) = *guard else { return };

    // Build offset slices for extract_ode_states
    let offset_slices: Option<Vec<(&[f32], &[f32])>> = mem_offsets.map(|m|
        m.offsets.iter().map(|(r, s)| (r.as_slice(), s.as_slice())).collect()
    );
    let mem_arg = offset_slices.as_deref();

    // Extract ODE states (truncate to block_size)
    let block_size = state.model.config.block_size;
    let start = if all_tokens.len() > block_size { all_tokens.len() - block_size } else { 0 };
    let ode_states = state.model.extract_ode_states(&all_tokens[start..]);

    // Merge into persistent memory using beta
    let beta = memory.config.beta;
    let w = 1.0 - beta;
    for (layer_idx, (r_avg, s_avg)) in ode_states.iter().enumerate() {
        if layer_idx >= memory.layers.len() { break; }
        let n = memory.n_bands.min(r_avg.len());
        for k in 0..n {
            memory.layers[layer_idx].r[k] = beta * memory.layers[layer_idx].r[k] + w * r_avg[k];
            memory.layers[layer_idx].s[k] = beta * memory.layers[layer_idx].s[k] + w * s_avg[k];
        }
    }
    memory.n_convos += 1;

    // Auto-save after each conversation
    if let Some(ref path) = state.memory_path {
        if let Err(e) = kerr_memory::file::save(path, memory) {
            eprintln!("  [memory save failed: {e}]");
        }
    }
}

async fn handle_non_streaming(
    state: Arc<AppState>,
    prompt_tokens: Vec<usize>,
    config: GenerationConfig,
    model_name: String,
    request_id: String,
    created: u64,
    prompt_len: usize,
    temperature: f32,
    top_p: f32,
) -> Response {
    let gen_start = std::time::Instant::now();
    let model = state.model.clone();
    let vocab = state.vocab.clone();
    let mem_offsets = get_memory_offsets(&state);
    let prompt_for_memory = prompt_tokens.clone();

    #[cfg(feature = "gpu")]
    let state_for_gpu = state.clone();
    #[cfg(not(feature = "gpu"))]
    let _ = &state; // suppress unused

    let result = tokio::task::spawn_blocking(move || {
        #[cfg(feature = "gpu")]
        {
            let mut guard = state_for_gpu.gpu.lock().unwrap();
            if let Some(ref mut gpu) = *guard {
                return inference::generate_with_forward(
                    &model, &prompt_tokens, &config, &vocab, mem_offsets.as_ref(),
                    |tokens, mem| {
                        crate::gpu_forward::forward_with_memory_gpu(&model, tokens, mem, gpu)
                    },
                );
            }
        }
        inference::generate(&model, &prompt_tokens, &config, &vocab, mem_offsets.as_ref())
    })
    .await
    .unwrap();

    // Telemetry
    log_request(prompt_len, result.tokens.len(), gen_start.elapsed().as_millis(),
                temperature, top_p, &result.text);

    // Update memory with full conversation (prompt + generated)
    let mem_offsets_for_update = get_memory_offsets(&state);
    let mut all_tokens = prompt_for_memory;
    all_tokens.extend(&result.tokens);
    update_memory(&state, &all_tokens, mem_offsets_for_update.as_ref());

    let response = ChatCompletionResponse {
        id: request_id,
        object: "chat.completion".to_string(),
        created,
        model: model_name,
        choices: vec![Choice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: result.text,
            },
            finish_reason: "length".to_string(),
        }],
        usage: Usage {
            prompt_tokens: prompt_len,
            completion_tokens: result.tokens.len(),
            total_tokens: prompt_len + result.tokens.len(),
        },
    };

    Json(response).into_response()
}

async fn handle_streaming(
    state: Arc<AppState>,
    prompt_tokens: Vec<usize>,
    config: GenerationConfig,
    model_name: String,
    request_id: String,
    created: u64,
    prompt_len: usize,
    temperature: f32,
    top_p: f32,
) -> Response {
    let gen_start = std::time::Instant::now();
    let model = state.model.clone();
    let vocab = state.vocab.clone();

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(32);

    // Send initial chunk with role
    let initial_chunk = ChatCompletionChunk {
        id: request_id.clone(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model_name.clone(),
        choices: vec![DeltaChoice {
            index: 0,
            delta: Delta {
                role: Some("assistant".to_string()),
                content: None,
            },
            finish_reason: None,
        }],
    };
    let _ = tx.send(Ok(Event::default()
        .data(serde_json::to_string(&initial_chunk).unwrap())))
        .await;

    let tx_clone = tx.clone();
    let req_id = request_id.clone();
    let mn = model_name.clone();
    let mem_offsets = get_memory_offsets(&state);

    // Note: memory accumulation not yet wired for streaming mode.
    // Non-streaming requests update memory after each conversation.
    tokio::task::spawn_blocking(move || {
        inference::generate_streaming(&model, &prompt_tokens, &config, &vocab, mem_offsets.as_ref(), |event| {
            let chunk = ChatCompletionChunk {
                id: req_id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: mn.clone(),
                choices: vec![DeltaChoice {
                    index: 0,
                    delta: Delta {
                        role: None,
                        content: Some(event.text),
                    },
                    finish_reason: if event.done { Some("length".to_string()) } else { None },
                }],
            };
            let json = serde_json::to_string(&chunk).unwrap();
            tx_clone.blocking_send(Ok(Event::default().data(json))).is_ok()
        });

        // Send [DONE] sentinel
        let _ = tx_clone.blocking_send(Ok(Event::default().data("[DONE]")));
    });

    Sse::new(ReceiverStream::new(rx))
        .into_response()
}

/// GET /v1/models
pub async fn handle_models(
    State(state): State<Arc<AppState>>,
) -> Json<ModelList> {
    Json(ModelList {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: state.config.model_name.clone(),
            object: "model".to_string(),
            created: 0,
            owned_by: "wave-engine".to_string(),
        }],
    })
}

/// GET /health
pub async fn handle_health(
    State(state): State<Arc<AppState>>,
) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        model: state.config.model_name.clone(),
        vocab_size: state.vocab.vocab_size,
        n_embd: state.model.config.n_embd(),
        n_layers: state.model.config.n_layers,
    })
}
