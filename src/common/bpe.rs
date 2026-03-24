//! BPE tokenizer — loads HuggingFace tokenizer.json for subword tokenization.
//!
//! Supports byte-level BPE (GPT-2/Llama/Qwen style). Loads vocabulary and merge
//! rules from a standard tokenizer.json file. Any text can be tokenized — the
//! byte-level mapping ensures no unknown bytes.

use std::collections::HashMap;
use std::fs;

/// Byte-Pair Encoding tokenizer loaded from a HuggingFace tokenizer.json file.
pub struct BpeTokenizer {
    /// Token string → token ID
    encoder: HashMap<String, usize>,
    /// Token ID → token string
    decoder: Vec<String>,
    /// Merge rules: (token_a, token_b) → merge rank (lower = higher priority)
    merges: HashMap<(String, String), usize>,
    /// Byte value → BPE unicode character
    byte_to_unicode: [char; 256],
    /// BPE unicode character → byte value
    unicode_to_byte: HashMap<char, u8>,
    /// Total vocabulary size
    pub vocab_size: usize,
}

// ─── Serde structs for tokenizer.json ─────────────────────────

#[derive(serde::Deserialize)]
struct TokenizerJson {
    model: BpeModel,
}

#[derive(serde::Deserialize)]
struct BpeModel {
    vocab: HashMap<String, usize>,
    merges: Vec<String>,
}

// ─── Implementation ───────────────────────────────────────────

impl BpeTokenizer {
    /// Load from a HuggingFace tokenizer.json file.
    pub fn from_file(path: &str) -> Self {
        let json_str = fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("Failed to read tokenizer file {path}: {e}"));

        let parsed: TokenizerJson = serde_json::from_str(&json_str)
            .unwrap_or_else(|e| panic!("Failed to parse tokenizer JSON: {e}"));

        let vocab_size = parsed.model.vocab.len();

        // Build decoder: ID → token string
        let mut decoder = vec![String::new(); vocab_size];
        for (token, &id) in &parsed.model.vocab {
            if id < vocab_size {
                decoder[id] = token.clone();
            }
        }

        // Build merge rank lookup
        let mut merges = HashMap::new();
        for (rank, merge_str) in parsed.model.merges.iter().enumerate() {
            if let Some(space_idx) = merge_str.find(' ') {
                let a = merge_str[..space_idx].to_string();
                let b = merge_str[space_idx + 1..].to_string();
                merges.insert((a, b), rank);
            }
        }

        let (byte_to_unicode, unicode_to_byte) = build_byte_to_unicode();

        println!("  BPE tokenizer: {} vocab, {} merges", vocab_size, merges.len());

        Self {
            encoder: parsed.model.vocab,
            decoder,
            merges,
            byte_to_unicode,
            unicode_to_byte,
            vocab_size,
        }
    }

    /// Encode text into token IDs using BPE.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut all_ids = Vec::new();

        // Pre-tokenize: split into chunks at whitespace boundaries
        for chunk in pre_tokenize(text) {
            let ids = self.bpe_encode_chunk(chunk.as_bytes());
            all_ids.extend(ids);
        }

        all_ids
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, tokens: &[usize]) -> String {
        // Concatenate BPE token strings
        let bpe_str: String = tokens.iter()
            .map(|&id| {
                if id < self.decoder.len() {
                    self.decoder[id].as_str()
                } else {
                    ""
                }
            })
            .collect();

        // Convert BPE unicode characters back to bytes
        let bytes: Vec<u8> = bpe_str.chars()
            .filter_map(|c| self.unicode_to_byte.get(&c).copied())
            .collect();

        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Apply BPE merges to a byte sequence (one pre-tokenized chunk).
    fn bpe_encode_chunk(&self, bytes: &[u8]) -> Vec<usize> {
        if bytes.is_empty() {
            return vec![];
        }

        // Convert bytes to BPE characters
        let mut tokens: Vec<String> = bytes.iter()
            .map(|&b| self.byte_to_unicode[b as usize].to_string())
            .collect();

        if tokens.len() == 1 {
            return match self.encoder.get(&tokens[0]) {
                Some(&id) => vec![id],
                None => vec![0],
            };
        }

        // Iteratively apply the lowest-rank merge
        loop {
            let mut best_rank = usize::MAX;
            let mut best_idx = None;

            for i in 0..tokens.len() - 1 {
                let pair = (tokens[i].clone(), tokens[i + 1].clone());
                if let Some(&rank) = self.merges.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_idx = Some(i);
                    }
                }
            }

            let Some(idx) = best_idx else { break };

            // Merge the pair
            let merged = format!("{}{}", tokens[idx], tokens[idx + 1]);
            tokens[idx] = merged;
            tokens.remove(idx + 1);
        }

        // Look up token IDs
        tokens.iter()
            .map(|t| *self.encoder.get(t).unwrap_or(&0))
            .collect()
    }
}

// ─── Pre-tokenization ────────────────────────────────────────

/// Split text into chunks for BPE processing.
/// Whitespace characters start a new chunk, attached as prefix to the following text.
/// "hello world" → ["hello", " world"]
fn pre_tokenize(text: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();

    for c in text.chars() {
        if (c == ' ' || c == '\n' || c == '\r' || c == '\t') && !current.is_empty() {
            chunks.push(current);
            current = String::new();
        }
        current.push(c);
    }

    if !current.is_empty() {
        chunks.push(current);
    }

    chunks
}

// ─── Byte ↔ Unicode mapping ──────────────────────────────────

/// Build the GPT-2 byte ↔ unicode character mapping.
///
/// Printable ASCII (33-126) and Latin-1 supplement (161-172, 174-255) map to
/// themselves. Remaining bytes (control chars, space, etc.) map to U+0100+.
/// This ensures every byte has a unique, visible unicode representation.
fn build_byte_to_unicode() -> ([char; 256], HashMap<char, u8>) {
    // Bytes that map directly to their unicode codepoint
    let mut direct: Vec<u8> = Vec::new();
    for b in 33..=126u8 { direct.push(b); }   // ASCII printable (! through ~)
    for b in 161..=172u8 { direct.push(b); }   // Latin-1 (¡ through ¬)
    for b in 174..=255u8 { direct.push(b); }   // Latin-1 (® through ÿ)

    let mut byte_to_unicode = ['\0'; 256];
    let mut unicode_to_byte = HashMap::new();

    // Direct mappings: byte value = unicode codepoint
    for &b in &direct {
        let ch = char::from(b);
        byte_to_unicode[b as usize] = ch;
        unicode_to_byte.insert(ch, b);
    }

    // Remaining bytes: map to U+0100, U+0101, ...
    let mut offset: u32 = 0;
    for b in 0..=255u8 {
        if byte_to_unicode[b as usize] == '\0' {
            let ch = char::from_u32(256 + offset).unwrap();
            byte_to_unicode[b as usize] = ch;
            unicode_to_byte.insert(ch, b);
            offset += 1;
        }
    }

    (byte_to_unicode, unicode_to_byte)
}
