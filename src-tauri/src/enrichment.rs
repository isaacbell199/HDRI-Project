//! Enrichment Pipeline for LLM Generation
//!
//! This module provides automatic prompt enrichment and output cleaning
//! to ensure high-quality, literary text generation.

use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashSet;

// ============================================================================
// Enrichment Constants
// ============================================================================

/// Density Instruction - Show, Don't Tell
const DENSITY_INSTRUCTION: &str = r#"
[WRITING DIRECTIVE]
Develop each action with precise sensory details:
- SIGHT: colors, lights, movements, facial expressions
- SOUND: noises, silences, tone of voice, rhythm
- TOUCH: textures, temperature, physical sensations
- SMELL: scents, ambient odors

GOLDEN RULE: ALWAYS replace emotion adjectives with physical manifestations.
- Instead of "he was frightened", describe: "his hands trembled", "his heart raced"
- Instead of "she was sad", describe: "her eyes filled with tears", "her voice broke"
"#;

/// Structure Constraint - Anti-clichés
const STRUCTURE_CONSTRAINT: &str = r#"
[LITERARY CONSTRAINTS]
ABSOLUTE PROHIBITIONS (AI clichés):
- Forbidden words: suddenly, abruptly, incredible, fantastic, extraordinary
- Forbidden words: a shiver ran through, without warning, at that precise moment
- Forbidden words: heart pounding, breath caught, eyes widened

RHYTHM AND VARIATION:
- Alternate between short sentences (impact) and long sentences (description)
- Avoid more than 3 sentences of the same length in a row
- Use sentence fragments for moments of intense action
"#;

/// Structure for multi-phrase modes (Story)
const MULTI_PHRASE_STRUCTURE: &str = r#"
[NARRATIVE STRUCTURE]
Compose your text in multiple sentences with:
1. An atmosphere/setup sentence
2. The main action or dialogue enriched
3. A reflection or internal reaction sentence
"#;

/// Structure for single phrase generation
const SINGLE_PHRASE_STRUCTURE: &str = r#"
[OUTPUT FORMAT]
Generate EXACTLY ONE SINGLE rich and evocative sentence.
This sentence must be complete, with appropriate final punctuation.
"#;

/// List of clichés to filter in post-processing
const CLICHE_PATTERNS: &[&str] = &[
    "suddenly,",
    "abruptly,",
    "without warning,",
    "at that precise moment,",
    "a shiver ran through",
    "heart pounding",
    "breath caught",
    "eyes widened",
    "all of a sudden,",
    "before anyone knew it,",
    "it was then that",
    "it goes without saying",
    "it must be said that",
    "it should not be forgotten that",
];

// ============================================================================
// Pre-compiled Regex Patterns (Performance Optimization)
// ============================================================================

/// Regex to collapse multiple whitespace into single space
static WHITESPACE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\s+").expect("Failed to compile WHITESPACE_RE")
});

/// Regex to match two or more consecutive periods (ellipsis normalization)
static ELLIPSIS_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\.{2,}").expect("Failed to compile ELLIPSIS_RE")
});

/// Regex to match spaces before punctuation
static SPACE_BEFORE_PUNCT_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\s+([.,!?;:])").expect("Failed to compile SPACE_BEFORE_PUNCT_RE")
});

/// Regex to match punctuation followed immediately by a letter (missing space)
static SPACE_AFTER_PUNCT_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"([.,!?;:])([A-Za-zÀ-ÖØ-öø-ÿ])").expect("Failed to compile SPACE_AFTER_PUNCT_RE")
});

/// Regex to match straight quotes for smart quote conversion
static QUOTE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#""([^"]+)""#).expect("Failed to compile QUOTE_RE")
});

/// Pre-compiled regex patterns for clichés at the start of a sentence
static CLICHE_START_PATTERNS: Lazy<Vec<Regex>> = Lazy::new(|| {
    CLICHE_PATTERNS
        .iter()
        .map(|cliche| {
            Regex::new(&format!(r"(?i)^{}\s*", regex::escape(cliche)))
                .expect("Failed to compile CLICHE_START_PATTERNS")
        })
        .collect()
});

/// Pre-compiled regex patterns for cliches anywhere in text
static CLICHE_ANYWHERE_PATTERNS: Lazy<Vec<Regex>> = Lazy::new(|| {
    CLICHE_PATTERNS
        .iter()
        .map(|cliche| {
            Regex::new(&format!(r"(?i){}\s*", regex::escape(cliche)))
                .expect("Failed to compile CLICHE_ANYWHERE_PATTERNS")
        })
        .collect()
});

/// Output language instruction - handles input/output language modes
const LANGUAGE_MODE_INSTRUCTION: &str = r#"
[LANGUAGE INSTRUCTIONS]
You will receive prompts and instructions in a specific input language.
You must generate your response in the specified output language.
Understand the input regardless of its language, but respond ONLY in the requested output language.
"#;

/// Common repetitive words to monitor
const REPETITIVE_WORDS: &[&str] = &[
    "then", "so", "next", "after", "as", "thus",
    "however", "nevertheless", "yet", "therefore", "for",
];

/// Words that should NOT be repeated in close proximity
const HIGH_FREQUENCY_WORDS: &[&str] = &[
    "the", "a", "an", "and", "or", "but", "in", "on", "at",
    "to", "for", "of", "with", "by", "from", "as", "is", "was",
    "are", "were", "been", "be", "have", "has", "had", "do",
    "does", "did", "will", "would", "could", "should", "may",
    "might", "must", "shall", "can", "need", "dare", "ought",
    "used", "it", "its", "this", "that", "these", "those",
];

/// Minimum distance between same word repetitions (in words)
const MIN_WORD_DISTANCE: usize = 10;

// ============================================================================
// Prompt Wrapping
// ============================================================================

/// Generation mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationMode {
    /// Story mode - allows multiple phrases
    Story,
    /// Dialogue mode - one phrase per line
    Dialogue,
    /// Description mode - one rich phrase
    Description,
    /// Action mode - one impact phrase
    Action,
    /// Continue mode - one continuation phrase
    Continue,
    /// Free mode - one creative phrase
    Free,
    /// Other mode - one default phrase
    Other,
}

impl GenerationMode {
    pub fn from_string(mode: &str) -> Self {
        match mode.to_lowercase().as_str() {
            "story" | "scene" => GenerationMode::Story,
            "dialogue" => GenerationMode::Dialogue,
            "describe" | "description" => GenerationMode::Description,
            "action" => GenerationMode::Action,
            "continue" | "continuation" => GenerationMode::Continue,
            "free" | "creative" => GenerationMode::Free,
            _ => GenerationMode::Other,
        }
    }

    /// Determines if this mode allows multiple phrases
    pub fn allows_multiple_phrases(&self) -> bool {
        matches!(self, GenerationMode::Story)
    }
}

/// Enrichment configuration
#[derive(Debug, Clone)]
pub struct EnrichmentConfig {
    /// Generation mode
    pub mode: GenerationMode,
    /// Start phrase (for Story mode)
    pub start_phrase: Option<String>,
    /// End phrase (for Story mode)
    pub end_phrase: Option<String>,
    /// Input language (language of user prompts): "en" or "fr"
    pub input_language: String,
    /// Output language (language of generated text): "en" or "fr"
    pub output_language: String,
    /// Custom style
    pub custom_style: Option<String>,
}

impl Default for EnrichmentConfig {
    fn default() -> Self {
        Self {
            mode: GenerationMode::Other,
            start_phrase: None,
            end_phrase: None,
            input_language: "en".to_string(),
            output_language: "en".to_string(),
            custom_style: None,
        }
    }
}

/// Wraps the user prompt with enrichment instructions
pub fn wrap_enriched_prompt(user_prompt: &str, config: &EnrichmentConfig) -> String {
    let mut enriched = String::new();

    // System header
    enriched.push_str("[SYSTEM - GENERATION INSTRUCTIONS]\n\n");

    // Density instructions (always present)
    enriched.push_str(DENSITY_INSTRUCTION);
    enriched.push_str("\n\n");

    // Structure constraints (always present)
    enriched.push_str(STRUCTURE_CONSTRAINT);
    enriched.push_str("\n\n");

    // Output structure based on mode
    let allows_multiple = config.mode.allows_multiple_phrases();

    if allows_multiple {
        // Check if we have start/end phrases
        let has_start = config.start_phrase.is_some();
        let has_end = config.end_phrase.is_some();

        if has_start || has_end {
            enriched.push_str("[CUSTOM STRUCTURE]\n");

            if has_start {
                enriched.push_str(&format!(
                    "Start your text AFTER this phrase: \"{}\"\n",
                    config.start_phrase.as_ref().unwrap()
                ));
            }

            if has_end {
                enriched.push_str(&format!(
                    "End your text BEFORE this phrase: \"{}\"\n",
                    config.end_phrase.as_ref().unwrap()
                ));
            }

            enriched.push_str("Generate the content between these two phrases with an appropriate word count to create a smooth and natural text.\n\n");
        } else {
            // Story mode without start/end phrases - multiple phrases
            enriched.push_str(MULTI_PHRASE_STRUCTURE);
            enriched.push_str("\n\n");
        }
    } else {
        // Standard mode - single phrase
        enriched.push_str(SINGLE_PHRASE_STRUCTURE);
        enriched.push_str("\n\n");
    }

    // Custom style if present
    if let Some(ref style) = config.custom_style {
        enriched.push_str(&format!("[CUSTOM STYLE]\n{}\n\n", style));
    }

    // Language mode instructions
    enriched.push_str(LANGUAGE_MODE_INSTRUCTION);
    enriched.push_str("\n");

    // Input language instruction
    let input_lang_name = match config.input_language.as_str() {
        "fr" => "French",
        "en" => "English",
        _ => &config.input_language
    };
    
    // Output language instruction
    let output_lang_name = match config.output_language.as_str() {
        "fr" => "French",
        "en" => "English",
        _ => &config.output_language
    };
    
    enriched.push_str(&format!(
        "[INPUT LANGUAGE]\nYour instructions and prompts are in {}.\n\n[OUTPUT LANGUAGE]\nGenerate the text in {}.\n\n",
        input_lang_name, output_lang_name
    ));

    // User prompt
    enriched.push_str("[USER PROMPT]\n");
    enriched.push_str(user_prompt);
    enriched.push_str("\n\n");

    // Final reminder for single phrase
    if !allows_multiple {
        enriched.push_str("[FINAL REMINDER]\n");
        enriched.push_str("⚠️ IMPORTANT: Generate EXACTLY ONE SINGLE SENTENCE. No more, no less.\n");
        enriched.push_str("The sentence must be literary, rich in sensory details, and end with a period.\n");
    }

    enriched
}

// ============================================================================
// Post-Processing
// ============================================================================

/// Cleans and improves the generated output
pub fn clean_output(text: &str, config: &EnrichmentConfig) -> String {
    let mut cleaned = text.to_string();

    // 1. Remove extra spaces (using pre-compiled regex)
    cleaned = WHITESPACE_RE.replace_all(&cleaned, " ").to_string();

    // 2. Remove clichés
    cleaned = remove_cliches(&cleaned);

    // 3. Normalize punctuation
    cleaned = normalize_punctuation(&cleaned);

    // 4. Advanced repetition detection and fixing
    cleaned = fix_repetitions(&cleaned);

    // 5. Fix phrase-level repetitions
    cleaned = fix_phrase_repetitions(&cleaned);

    // 6. For single phrase modes, ensure only one sentence
    if !config.mode.allows_multiple_phrases() {
        cleaned = ensure_single_phrase(&cleaned);
    }

    // 7. Final trim
    cleaned = cleaned.trim().to_string();

    // 8. Ensure it ends with a period
    if !cleaned.ends_with('.') && !cleaned.ends_with('!') && !cleaned.ends_with('?') && !cleaned.ends_with('…') {
        cleaned.push('.');
    }

    cleaned
}

/// Removes clichés from text
fn remove_cliches(text: &str) -> String {
    let mut result = text.to_string();

    // Use pre-compiled patterns for performance
    for (cliche, (start_re, anywhere_re)) in CLICHE_PATTERNS
        .iter()
        .zip(CLICHE_START_PATTERNS.iter().zip(CLICHE_ANYWHERE_PATTERNS.iter()))
    {
        let cliche_lower = cliche.to_lowercase();
        if result.to_lowercase().contains(&cliche_lower) {
            // For clichés at the beginning of a sentence
            if result.to_lowercase().starts_with(&cliche_lower) {
                // Remove and capitalize the first letter
                result = start_re.replace(&result, "").to_string();
                if !result.is_empty() {
                    let mut chars: Vec<char> = result.chars().collect();
                    chars[0] = chars[0].to_uppercase().next().unwrap_or(chars[0]);
                    result = chars.into_iter().collect();
                }
            } else {
                // Remove the cliché elsewhere
                result = anywhere_re.replace(&result, "").to_string();
            }
        }
    }

    result
}

/// Normalizes punctuation
fn normalize_punctuation(text: &str) -> String {
    let mut result = text.to_string();

    // Normalized ellipsis (using pre-compiled regex)
    result = ELLIPSIS_RE.replace_all(&result, "…").to_string();

    // Remove spaces before punctuation (using pre-compiled regex)
    result = SPACE_BEFORE_PUNCT_RE.replace_all(&result, "$1").to_string();

    // Add spaces after punctuation if missing (using pre-compiled regex)
    result = SPACE_AFTER_PUNCT_RE.replace_all(&result, "$1 $2").to_string();

    // Smart quotes (using pre-compiled regex)
    result = QUOTE_RE.replace_all(&result, "\"$1\"").to_string();

    result
}

/// Fixes word repetitions with advanced detection
fn fix_repetitions(text: &str) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 3 {
        return text.to_string();
    }

    let mut result = Vec::new();
    let mut recent_words = HashSet::new();
    let window_size = MIN_WORD_DISTANCE;

    // Track word positions for advanced repetition detection
    let mut word_positions: std::collections::HashMap<String, Vec<usize>> = std::collections::HashMap::new();

    for (i, &word) in words.iter().enumerate() {
        let word_lower = word.to_lowercase();
        let word_clean = word_lower.trim_matches(|c: char| !c.is_alphanumeric()).to_string();

        // Skip empty words
        if word_clean.is_empty() {
            result.push(word);
            continue;
        }

        // Check for immediate repetition (same word twice in a row)
        if i > 0 {
            let prev_word = words[i - 1].to_lowercase();
            let prev_clean = prev_word.trim_matches(|c: char| !c.is_alphanumeric()).to_string();
            if word_clean == prev_clean && word_clean.len() > 2 {
                // Skip immediate repetition
                continue;
            }
        }

        // Check for close proximity repetition of significant words
        let is_significant = !HIGH_FREQUENCY_WORDS.contains(&word_clean.as_str());
        let is_repetitive_connector = REPETITIVE_WORDS.contains(&word_clean.as_str());

        if is_repetitive_connector && recent_words.contains(&word_clean) {
            // Skip repetitive connector words that appear too close
            continue;
        }

        // For significant words, check if they appear too recently
        if is_significant && recent_words.contains(&word_clean) {
            // Only skip if it's a very close repetition (within 5 words)
            if let Some(positions) = word_positions.get(&word_clean) {
                if let Some(&last_pos) = positions.last() {
                    if i - last_pos < 5 {
                        // Skip very close repetition
                        continue;
                    }
                }
            }
        }

        result.push(word);

        // Track word position
        word_positions.entry(word_clean.clone()).or_insert_with(Vec::new).push(i);

        // Maintain the sliding window
        if i >= window_size {
            if let Some(old_word) = words.get(i - window_size) {
                let old_clean = old_word.to_lowercase()
                    .trim_matches(|c: char| !c.is_alphanumeric()).to_string();
                recent_words.remove(&old_clean);
            }
        }
        recent_words.insert(word_clean);
    }

    result.join(" ")
}

/// Detects and fixes repeated phrases (2-4 word sequences)
fn fix_phrase_repetitions(text: &str) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 8 {
        return text.to_string();
    }

    // Build n-grams and detect repetitions
    let mut result = words.clone();
    let mut detected_ranges: Vec<(usize, usize)> = Vec::new();

    // Check for 2-4 word phrase repetitions
    for n in 2..=4 {
        let mut phrase_positions: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

        for i in 0..=words.len().saturating_sub(n) {
            let phrase = words[i..i+n].join(" ").to_lowercase();
            let phrase_clean = phrase.trim_matches(|c: char| !c.is_alphanumeric() && c != ' ').to_string();

            if phrase_clean.len() < 4 {
                continue; // Skip very short phrases
            }

            if let Some(&prev_pos) = phrase_positions.get(&phrase_clean) {
                // Found a repetition - mark for potential removal
                let distance = i - prev_pos;
                if distance < 10 && distance > n {
                    // Close repetition detected
                    detected_ranges.push((i, i + n));
                }
            } else {
                phrase_positions.insert(phrase_clean, i);
            }
        }
    }

    // Remove detected repetitions (in reverse order to maintain indices)
    detected_ranges.sort_by(|a, b| b.0.cmp(&a.0));
    for (start, _end) in detected_ranges {
        // Don't remove, just log for now - aggressive removal can break text
        // In future, could replace with synonym or restructure
        log::debug!("Detected phrase repetition at position {}", start);
    }

    result.join(" ")
}

/// Ensures the text contains only one sentence
fn ensure_single_phrase(text: &str) -> String {
    // Find the first complete sentence
    let sentence_enders = ['.', '!', '?', '…'];

    // Find the end of the first sentence
    let mut end_pos = text.len();
    let mut found_end = false;

    for (i, c) in text.char_indices() {
        if sentence_enders.contains(&c) {
            end_pos = i + c.len_utf8();
            found_end = true;
            break;
        }
    }

    if found_end {
        text[..end_pos].to_string()
    } else {
        // If no period, add one
        format!("{}.", text.trim())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wrap_enriched_prompt_single_phrase() {
        let config = EnrichmentConfig {
            mode: GenerationMode::Continue,
            ..Default::default()
        };

        let result = wrap_enriched_prompt("Continue the story", &config);

        // Check for the single phrase instruction
        assert!(result.contains("EXACTLY ONE SINGLE SENTENCE") || result.contains("EXACTLY ONE SINGLE rich and evocative sentence"));
    }

    #[test]
    fn test_wrap_enriched_prompt_story_mode() {
        let config = EnrichmentConfig {
            mode: GenerationMode::Story,
            start_phrase: Some("The sun was setting.".to_string()),
            end_phrase: Some("She smiled.".to_string()),
            ..Default::default()
        };

        let result = wrap_enriched_prompt("Describe the scene", &config);

        assert!(result.contains("CUSTOM STRUCTURE"));
        assert!(result.contains("The sun was setting"));
        assert!(result.contains("She smiled"));
    }

    #[test]
    fn test_clean_output_single_phrase() {
        let config = EnrichmentConfig {
            mode: GenerationMode::Continue,
            ..Default::default()
        };

        let input = "Night fell over the city. Lights came on one by one.";
        let result = clean_output(input, &config);

        // Should keep only the first sentence
        assert_eq!(result, "Night fell over the city.");
    }

    #[test]
    fn test_clean_output_story_mode() {
        let config = EnrichmentConfig {
            mode: GenerationMode::Story,
            ..Default::default()
        };

        let input = "Night fell. Lights shone. The city awakened.";
        let result = clean_output(input, &config);

        // Should keep all sentences
        assert!(result.contains("Night") && result.contains("Lights") && result.contains("city"));
    }

    #[test]
    fn test_remove_cliches() {
        let input = "Suddenly, she understood that everything had changed.";
        let result = remove_cliches(input);

        assert!(!result.to_lowercase().contains("suddenly"));
    }

    #[test]
    fn test_normalize_punctuation() {
        let input = "She said:\"Hello!\" ";
        let result = normalize_punctuation(input);

        assert!(result.contains("\""));
    }
}
