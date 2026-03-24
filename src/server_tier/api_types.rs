//! OpenAI-compatible API types — pure serde structs, no logic.

use serde::{Deserialize, Serialize};

// ─── Chat Completion Request ─────────────────────────────────

#[derive(Deserialize)]
pub struct ChatCompletionRequest {
    #[allow(dead_code)]
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
}

fn default_temperature() -> f32 { 0.8 }
fn default_top_p() -> f32 { 1.0 }
fn default_max_tokens() -> usize { 256 }

#[derive(Deserialize, Serialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

// ─── Chat Completion Response ────────────────────────────────

#[derive(Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Serialize)]
pub struct Choice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

// ─── Streaming Types ─────────────────────────────────────────

#[derive(Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<DeltaChoice>,
}

#[derive(Serialize)]
pub struct DeltaChoice {
    pub index: usize,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

// ─── Models Endpoint ─────────────────────────────────────────

#[derive(Serialize)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

#[derive(Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

// ─── Health Endpoint ─────────────────────────────────────────

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub model: String,
    pub vocab_size: usize,
    pub n_embd: usize,
    pub n_layers: usize,
}

// ─── Error Response ──────────────────────────────────────────

#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Serialize)]
pub struct ErrorDetail {
    pub message: String,
    pub r#type: String,
}
