use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Response from the AI model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelResponse {
    /// The generated text response
    pub content: String,
    /// Metadata about the response (tokens used, etc.)
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Evaluation result for a single response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResult {
    /// Scores for each evaluation category (0.0 to 1.0)
    pub scores: HashMap<String, f64>,
    /// Detailed feedback from the evaluator
    pub feedback: String,
    /// Raw evaluation response
    pub raw_response: String,
}

/// Complete result for a single prompt-response-evaluation cycle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptResult {
    /// Original prompt
    pub prompt: String,
    /// Model's response
    pub response: ModelResponse,
    /// Evaluation of the response
    pub evaluation: EvaluationResult,
}

/// Statistics calculated across multiple results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Statistics {
    /// Mean scores for each category
    pub mean: HashMap<String, f64>,
    /// Median scores for each category
    pub median: HashMap<String, f64>,
    /// Mode scores for each category (most frequent score)
    pub mode: HashMap<String, f64>,
}

/// Final results containing statistics and individual results
#[derive(Debug, Serialize, Deserialize)]
pub struct FinalResults {
    /// Aggregated statistics
    pub statistics: Statistics,
    /// Individual prompt results
    pub results: Vec<PromptResult>,
}

