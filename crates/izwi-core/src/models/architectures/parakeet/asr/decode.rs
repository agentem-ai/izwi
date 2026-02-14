use std::fs;
use std::path::Path;

use crate::error::{Error, Result};

pub fn load_tokenizer_vocab(path: &Path) -> Result<Vec<String>> {
    let raw = fs::read_to_string(path).map_err(|e| {
        Error::ModelLoadError(format!(
            "Failed to read tokenizer vocab {}: {}",
            path.display(),
            e
        ))
    })?;

    let mut vocab = Vec::new();
    for line in raw.lines() {
        let token = line.split('\t').next().unwrap_or("").trim();
        if token.is_empty() {
            continue;
        }
        vocab.push(token.to_string());
    }

    if vocab.is_empty() {
        return Err(Error::ModelLoadError(format!(
            "Tokenizer vocab at {} is empty",
            path.display()
        )));
    }

    Ok(vocab)
}

pub fn decode_tokens(ids: &[usize], vocab: &[String]) -> String {
    let mut out = String::new();

    for &id in ids {
        let Some(tok) = vocab.get(id) else {
            continue;
        };

        if tok == "<unk>" || tok == "<pad>" {
            continue;
        }
        if tok.starts_with("<|") && tok.ends_with("|>") {
            continue;
        }

        if tok.starts_with('▁') {
            let piece = tok.trim_start_matches('▁');
            if !out.is_empty() && !out.ends_with(' ') {
                out.push(' ');
            }
            out.push_str(piece);
            continue;
        }

        if let Some(piece) = tok.strip_prefix("##") {
            out.push_str(piece);
            continue;
        }

        // SentencePiece tokens without a leading ▁ continue the current word.
        out.push_str(tok);
    }

    normalize_text(out)
}

fn normalize_text(mut s: String) -> String {
    s = s.split_whitespace().collect::<Vec<_>>().join(" ");

    for punct in [".", ",", "!", "?", ":", ";"] {
        s = s.replace(&format!(" {punct}"), punct);
    }

    s.trim().to_string()
}
