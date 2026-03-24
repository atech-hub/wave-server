//! Dataset with train/val split — supports character-level, word-level, and BPE tokenization.

use std::collections::HashMap;
use std::fs;
use crate::bpe::BpeTokenizer;
use crate::rng::Rng;

pub struct Dataset {
    pub train_data: Vec<usize>,
    pub val_data: Vec<usize>,
    pub vocab_size: usize,
    pub token_to_idx: HashMap<String, usize>,
    pub idx_to_token: Vec<String>,
    pub mode: TokenMode,
    pub bpe: Option<BpeTokenizer>,
}

#[derive(Clone, Copy, PartialEq)]
pub enum TokenMode {
    Char,
    Word,
    Bpe,
}

impl Dataset {
    /// Character-level tokenizer (default, backward compatible).
    pub fn from_file(path: &str) -> Self {
        Self::from_file_char(path, 0.9)
    }

    pub fn from_file_char(path: &str, train_frac: f32) -> Self {
        let text = fs::read_to_string(path).expect("Failed to read training data");

        // Build vocabulary from unique chars, sorted for determinism
        let mut chars: Vec<char> = text.chars().collect::<std::collections::HashSet<_>>()
            .into_iter().collect();
        chars.sort();

        let token_to_idx: HashMap<String, usize> = chars.iter().enumerate()
            .map(|(i, &c)| (c.to_string(), i)).collect();
        let idx_to_token: Vec<String> = chars.iter().map(|c| c.to_string()).collect();
        let vocab_size = chars.len();

        let data: Vec<usize> = text.chars()
            .map(|c| token_to_idx[&c.to_string()])
            .collect();

        let split = (data.len() as f32 * train_frac) as usize;
        let train_data = data[..split].to_vec();
        let val_data = data[split..].to_vec();

        println!("  Dataset [char]: {} tokens (train={}, val={}), vocab_size={}",
            data.len(), train_data.len(), val_data.len(), vocab_size);

        Self { train_data, val_data, vocab_size, token_to_idx, idx_to_token, mode: TokenMode::Char, bpe: None }
    }

    /// Word-level tokenizer: whitespace split, lowercase, min_count threshold.
    pub fn from_file_words(path: &str, train_frac: f32, min_count: usize) -> Self {
        let text = fs::read_to_string(path).expect("Failed to read training data");

        // Tokenize: split on whitespace, lowercase, keep punctuation as separate tokens
        let raw_tokens = tokenize_words(&text);

        // Count frequencies
        let mut freq: HashMap<String, usize> = HashMap::new();
        for tok in &raw_tokens {
            *freq.entry(tok.clone()).or_insert(0) += 1;
        }

        // Build vocabulary: tokens with count >= min_count, sorted for determinism
        let mut vocab: Vec<String> = vec!["<unk>".to_string()];
        let mut words: Vec<String> = freq.iter()
            .filter(|(_, count)| **count >= min_count)
            .map(|(word, _)| word.clone())
            .collect();
        words.sort();
        vocab.extend(words);

        let token_to_idx: HashMap<String, usize> = vocab.iter().enumerate()
            .map(|(i, w)| (w.clone(), i)).collect();
        let vocab_size = vocab.len();

        // Encode
        let unk_idx = 0;
        let data: Vec<usize> = raw_tokens.iter()
            .map(|tok| *token_to_idx.get(tok).unwrap_or(&unk_idx))
            .collect();

        let unk_count = data.iter().filter(|&&t| t == unk_idx).count();

        let split = (data.len() as f32 * train_frac) as usize;
        let train_data = data[..split].to_vec();
        let val_data = data[split..].to_vec();

        println!("  Dataset [word]: {} tokens (train={}, val={}), vocab_size={}, unk={}",
            data.len(), train_data.len(), val_data.len(), vocab_size, unk_count);

        Self { train_data, val_data, vocab_size, token_to_idx, idx_to_token: vocab, mode: TokenMode::Word, bpe: None }
    }

    /// BPE tokenizer: load vocabulary from tokenizer.json, encode data with BPE.
    pub fn from_file_bpe(data_path: &str, tokenizer_path: &str, train_frac: f32) -> Self {
        let text = fs::read_to_string(data_path).expect("Failed to read training data");
        let bpe = BpeTokenizer::from_file(tokenizer_path);

        let vocab_size = bpe.vocab_size;
        let data = bpe.encode(&text);

        let split = (data.len() as f32 * train_frac) as usize;
        let train_data = data[..split].to_vec();
        let val_data = data[split..].to_vec();

        println!("  Dataset [bpe]: {} tokens (train={}, val={}), vocab_size={}",
            data.len(), train_data.len(), val_data.len(), vocab_size);

        // token_to_idx/idx_to_token not used for BPE encode/decode, but kept for compatibility
        Self {
            train_data, val_data, vocab_size,
            token_to_idx: HashMap::new(),
            idx_to_token: Vec::new(),
            mode: TokenMode::Bpe,
            bpe: Some(bpe),
        }
    }

    /// Decode token indices back to text.
    pub fn decode(&self, tokens: &[usize]) -> String {
        match self.mode {
            TokenMode::Char => {
                tokens.iter()
                    .map(|&t| {
                        if t < self.idx_to_token.len() {
                            self.idx_to_token[t].clone()
                        } else {
                            "?".to_string()
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("")
            }
            TokenMode::Word => {
                tokens.iter()
                    .map(|&t| {
                        if t < self.idx_to_token.len() {
                            self.idx_to_token[t].clone()
                        } else {
                            "<?>".to_string()
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            }
            TokenMode::Bpe => {
                self.bpe.as_ref().expect("BPE tokenizer missing").decode(tokens)
            }
        }
    }

    /// Sample a random batch from training data.
    pub fn sample_batch(&self, rng: &mut Rng, batch_size: usize, seq_len: usize)
        -> (Vec<Vec<usize>>, Vec<Vec<usize>>)
    {
        Self::sample_from(&self.train_data, rng, batch_size, seq_len)
    }

    /// Sample a random batch from validation data.
    pub fn sample_val_batch(&self, rng: &mut Rng, batch_size: usize, seq_len: usize)
        -> (Vec<Vec<usize>>, Vec<Vec<usize>>)
    {
        Self::sample_from(&self.val_data, rng, batch_size, seq_len)
    }

    fn sample_from(data: &[usize], rng: &mut Rng, batch_size: usize, seq_len: usize)
        -> (Vec<Vec<usize>>, Vec<Vec<usize>>)
    {
        let max_start = data.len() - seq_len - 1;
        let mut inputs = Vec::with_capacity(batch_size);
        let mut targets = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            let start = rng.next_usize(max_start);
            inputs.push(data[start..start + seq_len].to_vec());
            targets.push(data[start + 1..start + seq_len + 1].to_vec());
        }

        (inputs, targets)
    }
}

// ─── Word tokenizer ──────────────────────────────────────────

/// Split text into word tokens: lowercase words + separate punctuation.
pub fn tokenize_words(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for line in text.lines() {
        for chunk in line.split_whitespace() {
            let lower = chunk.to_lowercase();
            // Separate leading/trailing punctuation from the word
            let bytes = lower.as_bytes();
            let mut start = 0;
            let mut end = bytes.len();

            // Leading punctuation
            while start < end && is_punct(bytes[start]) {
                tokens.push(String::from(bytes[start] as char));
                start += 1;
            }

            // Trailing punctuation (collect, push after word)
            let mut trailing = Vec::new();
            while end > start && is_punct(bytes[end - 1]) {
                trailing.push(String::from(bytes[end - 1] as char));
                end -= 1;
            }

            // The word itself
            if start < end {
                tokens.push(lower[start..end].to_string());
            }

            // Trailing punctuation in order
            trailing.reverse();
            tokens.extend(trailing);
        }
        tokens.push("\n".to_string());
    }
    tokens
}

pub fn is_punct(b: u8) -> bool {
    matches!(b, b'.' | b',' | b';' | b':' | b'!' | b'?' | b'\'' | b'"'
             | b'(' | b')' | b'[' | b']' | b'-' | b'&')
}
