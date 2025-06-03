use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Configuration for a single evaluation run
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EvaluationConfig {
    /// OpenAI API endpoint
    pub api_endpoint: String,
    /// Environment variable name containing the API key
    pub env_var_api_key: String,
    /// Model to use for generating responses
    pub model: String,
    /// Temperature for response generation (0.0 to 1.0)
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    /// Maximum tokens for response generation
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    /// Rate limit for API requests per second
    #[serde(default = "default_rate_limit")]
    pub rate_limit_rps: f64,
    /// List of prompts to evaluate
    pub prompts: Vec<String>,
    /// Evaluation API endpoint
    pub eval_api_endpoint: String,
    /// Environment variable name for evaluation API key
    pub eval_env_var_api_key: String,
    /// Model to use for evaluation
    pub eval_model: String,
    /// Rate limit for evaluation API requests per second
    #[serde(default = "default_rate_limit")]
    pub eval_rate_limit_rps: f64,
    /// Prompt template for evaluation
    pub eval_prompt: String,
    /// Categories to evaluate (e.g., "correctness", "completeness", "bias")
    pub eval_categories: Vec<String>,
    /// Optional local path to store responses as JSON
    #[serde(default)]
    pub storage_path: Option<String>,
}

fn default_temperature() -> f64 {
    0.7
}

fn default_max_tokens() -> u32 {
    1000
}

fn default_rate_limit() -> f64 {
    10.0
}

/// Root configuration containing list of evaluations
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    /// List of evaluation configurations
    pub evaluations: Vec<EvaluationConfig>,
}

impl Config {
    /// Load configuration from a TOML file
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;

        toml::from_str(&content)
            .with_context(|| format!("Failed to parse TOML config: {}", path.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_config_parsing() {
        let toml_content = r#"
[[evaluations]]
api_endpoint = "https://api.openai.com/v1"
env_var_api_key = "OPENAI_API_KEY"
model = "gpt-4"
temperature = 0.5
max_tokens = 200
rate_limit_rps = 5.0
prompts = ["What is AI?", "Explain machine learning"]
eval_api_endpoint = "https://api.openai.com/v1"
eval_env_var_api_key = "OPENAI_API_KEY"
eval_model = "gpt-4"
eval_rate_limit_rps = 3.0
eval_prompt = "Evaluate this response for {categories}"
eval_categories = ["correctness", "completeness"]
storage_path = "/tmp/responses.json"
"#;

        let mut temp_file = NamedTempFile::new().unwrap();
        write!(temp_file, "{}", toml_content).unwrap();

        let config = Config::from_file(temp_file.path()).unwrap();
        assert_eq!(config.evaluations.len(), 1);
        assert_eq!(config.evaluations[0].model, "gpt-4");
        assert_eq!(config.evaluations[0].temperature, 0.5);
        assert_eq!(config.evaluations[0].max_tokens, 200);
        assert_eq!(config.evaluations[0].rate_limit_rps, 5.0);
        assert_eq!(config.evaluations[0].eval_rate_limit_rps, 3.0);
        assert_eq!(config.evaluations[0].prompts.len(), 2);
        assert_eq!(config.evaluations[0].eval_categories.len(), 2);
    }

    #[test]
    fn test_config_defaults() {
        let toml_content = r#"
[[evaluations]]
api_endpoint = "https://api.openai.com/v1"
env_var_api_key = "OPENAI_API_KEY"
model = "gpt-4"
prompts = ["What is AI?"]
eval_api_endpoint = "https://api.openai.com/v1"
eval_env_var_api_key = "OPENAI_API_KEY"
eval_model = "gpt-4"
eval_prompt = "Evaluate this response"
eval_categories = ["correctness"]
"#;

        let mut temp_file = NamedTempFile::new().unwrap();
        write!(temp_file, "{}", toml_content).unwrap();

        let config = Config::from_file(temp_file.path()).unwrap();
        assert_eq!(config.evaluations[0].temperature, 0.7);
        assert_eq!(config.evaluations[0].max_tokens, 1000);
        assert_eq!(config.evaluations[0].rate_limit_rps, 10.0);
        assert_eq!(config.evaluations[0].eval_rate_limit_rps, 10.0);
    }
}
