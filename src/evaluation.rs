use crate::config::EvaluationConfig;
use crate::models::{EvaluationResult, ModelResponse, Statistics};
use anyhow::{Context, Result};
use async_openai::{Client, config::OpenAIConfig, types::CreateChatCompletionRequestArgs};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::time::sleep;

/// evaluator for evaluating AI model responses with rate limiting
pub struct Evaluator {
    /// Last request time for main API endpoint
    last_api_request: Option<Instant>,
    /// Last request time for evaluation API endpoint
    last_eval_request: Option<Instant>,
}

impl Evaluator {
    /// Create a new evaluator
    pub fn new() -> Self {
        Self {
            last_api_request: None,
            last_eval_request: None,
        }
    }

    /// Enforce rate limiting for API requests
    async fn enforce_rate_limit(last_request: &mut Option<Instant>, rate_limit_rps: f64) {
        if rate_limit_rps <= 0.0 {
            return;
        }

        let min_interval = Duration::from_secs_f64(1.0 / rate_limit_rps);

        if let Some(last_time) = *last_request {
            let elapsed = last_time.elapsed();
            if elapsed < min_interval {
                let sleep_duration = min_interval - elapsed;
                sleep(sleep_duration).await;
            }
        }

        *last_request = Some(Instant::now());
    }

    /// Generate a response using the specified model and prompt
    pub async fn generate_response(
        &mut self,
        config: &EvaluationConfig,
        prompt: &str,
    ) -> Result<ModelResponse> {
        Self::enforce_rate_limit(&mut self.last_api_request, config.rate_limit_rps).await;

        let client = self.create_main_client(config)?;
        let request = self.build_main_request(config, prompt)?;
        let response = self.execute_main_request(&client, request).await?;
        
        Ok(self.extract_model_response(response))
    }

    /// Create the OpenAI client for the main API
    fn create_main_client(&self, config: &EvaluationConfig) -> Result<Client<OpenAIConfig>> {
        let api_key = std::env::var(&config.env_var_api_key)
            .with_context(|| format!("Environment variable {} not found", config.env_var_api_key))?;

        let openai_config = OpenAIConfig::new()
            .with_api_key(api_key)
            .with_api_base(&config.api_endpoint);

        Ok(Client::with_config(openai_config))
    }

    /// Build the chat completion request for the main API
    fn build_main_request(
        &self,
        config: &EvaluationConfig,
        prompt: &str,
    ) -> Result<async_openai::types::CreateChatCompletionRequest> {
        let system_message = async_openai::types::ChatCompletionRequestSystemMessageArgs::default()
            .content(config.system_prompt.clone())
            .build()
            .context("Failed to build system message")?
            .into();

        let user_message = async_openai::types::ChatCompletionRequestUserMessageArgs::default()
            .content(prompt.to_string())
            .build()
            .context("Failed to build user message")?
            .into();

        CreateChatCompletionRequestArgs::default()
            .model(&config.model)
            .messages([system_message, user_message])
            .temperature(config.temperature as f32)
            .max_tokens(config.max_tokens as u16)
            .build()
            .context("Failed to build chat completion request")
    }

    /// Execute the main API request
    async fn execute_main_request(
        &self,
        client: &Client<OpenAIConfig>,
        request: async_openai::types::CreateChatCompletionRequest,
    ) -> Result<async_openai::types::CreateChatCompletionResponse> {
        client
            .chat()
            .create(request)
            .await
            .context("Failed to generate response")
    }

    /// Extract ModelResponse from the API response
    fn extract_model_response(
        &self,
        response: async_openai::types::CreateChatCompletionResponse,
    ) -> ModelResponse {
        let content = match response.choices.first() {
            Some(choice) => match &choice.message.content {
                Some(content) => content.clone(),
                None => String::new(),
            },
            None => String::new(),
        };

        let mut metadata = HashMap::new();
        if let Some(usage) = response.usage {
            metadata.insert("prompt_tokens".to_string(), json!(usage.prompt_tokens));
            metadata.insert("completion_tokens".to_string(), json!(usage.completion_tokens));
            metadata.insert("total_tokens".to_string(), json!(usage.total_tokens));
        }

        ModelResponse { content, metadata }
    }

    /// Evaluate a response using the evaluation model
    pub async fn evaluate_response(
        &mut self,
        config: &EvaluationConfig,
        prompt: &str,
        response: &str,
    ) -> Result<EvaluationResult> {
        Self::enforce_rate_limit(&mut self.last_eval_request, config.eval_rate_limit_rps).await;

        let client = self.create_eval_client(config)?;
        let full_eval_prompt = self.build_eval_prompt(config, prompt, response);
        let request = self.build_eval_request(config, &full_eval_prompt)?;
        let eval_response = self.execute_eval_request(&client, request).await?;
        let eval_content = self.extract_eval_content(eval_response);

        self.parse_evaluation_response(&eval_content, &config.eval_categories)
    }

    /// Create the OpenAI client for the evaluation API
    fn create_eval_client(&self, config: &EvaluationConfig) -> Result<Client<OpenAIConfig>> {
        let eval_api_key = std::env::var(&config.eval_env_var_api_key)
            .with_context(|| format!("Environment variable {} not found", config.eval_env_var_api_key))?;

        let openai_config = OpenAIConfig::new()
            .with_api_key(eval_api_key)
            .with_api_base(&config.eval_api_endpoint);

        Ok(Client::with_config(openai_config))
    }

    /// Build the evaluation prompt
    fn build_eval_prompt(&self, config: &EvaluationConfig, prompt: &str, response: &str) -> String {
        let categories_str = config.eval_categories.join(", ");
        let eval_prompt = config.eval_prompt.replace("{categories}", &categories_str);

        format!(
            "{}\n\nOriginal Prompt: {}\nResponse to Evaluate: {}\n\nPlease provide scores (0.0 to 1.0) for each category and detailed feedback. Return as JSON with 'scores' and 'feedback' fields.",
            eval_prompt, prompt, response
        )
    }

    /// Build the evaluation request
    fn build_eval_request(
        &self,
        config: &EvaluationConfig,
        full_eval_prompt: &str,
    ) -> Result<async_openai::types::CreateChatCompletionRequest> {
        let system_message = async_openai::types::ChatCompletionRequestSystemMessageArgs::default()
            .content(config.eval_system_prompt.clone())
            .build()
            .context("Failed to build eval system message")?
            .into();

        let user_message = async_openai::types::ChatCompletionRequestUserMessageArgs::default()
            .content(full_eval_prompt.to_string())
            .build()
            .context("Failed to build eval user message")?
            .into();

        CreateChatCompletionRequestArgs::default()
            .model(&config.eval_model)
            .messages([system_message, user_message])
            .temperature(0.1)
            .build()
            .context("Failed to build eval completion request")
    }

    /// Execute the evaluation API request
    async fn execute_eval_request(
        &self,
        client: &Client<OpenAIConfig>,
        request: async_openai::types::CreateChatCompletionRequest,
    ) -> Result<async_openai::types::CreateChatCompletionResponse> {
        client
            .chat()
            .create(request)
            .await
            .context("Failed to generate evaluation response")
    }

    /// Extract content from evaluation response
    fn extract_eval_content(
        &self,
        eval_response: async_openai::types::CreateChatCompletionResponse,
    ) -> String {
        match eval_response.choices.first() {
            Some(choice) => match &choice.message.content {
                Some(content) => content.clone(),
                None => String::new(),
            },
            None => String::new(),
        }
    }

    /// Parse the evaluation response JSON
    fn parse_evaluation_response(
        &self,
        response: &str,
        categories: &[String],
    ) -> Result<EvaluationResult> {
        let parsed = self.parse_json_response(response)?;
        let scores = self.extract_scores(&parsed, categories);
        let feedback = self.extract_feedback(&parsed);

        Ok(EvaluationResult {
            scores,
            feedback,
            raw_response: response.to_string(),
        })
    }

    /// Parse JSON from the response, handling embedded JSON
    fn parse_json_response(&self, response: &str) -> Result<Value> {
        match serde_json::from_str(response) {
            Ok(parsed) => Ok(parsed),
            Err(_) => self.try_extract_embedded_json(response),
        }
    }

    /// Try to extract JSON that might be embedded in text
    fn try_extract_embedded_json(&self, response: &str) -> Result<Value> {
        match response.find('{') {
            Some(start) => match response.rfind('}') {
                Some(end) => serde_json::from_str(&response[start..=end])
                    .context("Failed to parse extracted JSON"),
                None => anyhow::bail!("Found opening brace but no closing brace in response"),
            },
            None => anyhow::bail!("No JSON found in response"),
        }
    }

    /// Extract scores from parsed JSON
    fn extract_scores(&self, parsed: &Value, categories: &[String]) -> HashMap<String, f64> {
        let mut scores = HashMap::new();

        match parsed.get("scores").and_then(|s| s.as_object()) {
            Some(scores_obj) => {
                for category in categories {
                    let score = match scores_obj.get(category).and_then(|s| s.as_f64()) {
                        Some(score) => score.clamp(0.0, 1.0),
                        None => 0.0,
                    };
                    scores.insert(category.clone(), score);
                }
            }
            None => {
                // Don't populate with default values when no scores object exists
                // Leave the HashMap empty
            }
        }

        scores
    }

    /// Extract feedback from parsed JSON
    fn extract_feedback(&self, parsed: &Value) -> String {
        match parsed.get("feedback").and_then(|f| f.as_str()) {
            Some(feedback) => feedback.to_string(),
            None => "No feedback provided".to_string(),
        }
    }

    /// Calculate statistics across multiple evaluation results
    pub fn calculate_statistics(
        &self,
        results: &[&EvaluationResult],
        categories: &[String],
    ) -> Statistics {
        let mut mean = HashMap::new();
        let mut median = HashMap::new();
        let mut mode = HashMap::new();

        for category in categories {
            let scores = self.collect_category_scores(results, category);
            
            if scores.is_empty() {
                self.insert_zero_stats(category, &mut mean, &mut median, &mut mode);
                continue;
            }

            mean.insert(category.clone(), self.calculate_mean(&scores));
            median.insert(category.clone(), self.calculate_median(&scores));
            mode.insert(category.clone(), self.calculate_mode(&scores));
        }

        Statistics { mean, median, mode }
    }

    /// Collect scores for a specific category
    fn collect_category_scores(&self, results: &[&EvaluationResult], category: &str) -> Vec<f64> {
        results
            .iter()
            .filter_map(|r| r.scores.get(category))
            .copied()
            .collect()
    }

    /// Insert zero values for all statistics
    fn insert_zero_stats(
        &self,
        category: &str,
        mean: &mut HashMap<String, f64>,
        median: &mut HashMap<String, f64>,
        mode: &mut HashMap<String, f64>,
    ) {
        mean.insert(category.to_string(), 0.0);
        median.insert(category.to_string(), 0.0);
        mode.insert(category.to_string(), 0.0);
    }

    /// Calculate mean of scores
    fn calculate_mean(&self, scores: &[f64]) -> f64 {
        let sum: f64 = scores.iter().sum();
        sum / scores.len() as f64
    }

    /// Calculate median of scores
    fn calculate_median(&self, scores: &[f64]) -> f64 {
        let mut sorted_scores = scores.to_vec();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let mid = sorted_scores.len() / 2;
        if sorted_scores.len() % 2 == 0 {
            (sorted_scores[mid - 1] + sorted_scores[mid]) / 2.0
        } else {
            sorted_scores[mid]
        }
    }

    /// Calculate mode of scores (most frequent value, rounded to 1 decimal place)
    fn calculate_mode(&self, scores: &[f64]) -> f64 {
        let mut frequency = HashMap::new();
        
        for &score in scores {
            let rounded = ((score * 10.0).round() as i32) as f64 / 10.0;
            *frequency.entry(rounded.to_bits()).or_insert(0) += 1;
        }

        match frequency.iter().max_by_key(|&(_, count)| count) {
            Some((&bits, _)) => f64::from_bits(bits),
            None => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EvaluationConfig;
    use std::time::Duration;
    use tokio::time::Instant as TokioInstant;

    fn create_test_config() -> EvaluationConfig {
        EvaluationConfig {
            title: "test".to_string(),
            api_endpoint: "https://api.openai.com/v1".to_string(),
            env_var_api_key: "TEST_API_KEY".to_string(),
            model: "gpt-4".to_string(),
            temperature: 0.7,
            max_tokens: 1000,
            rate_limit_rps: 10.0,
            prompts: vec!["Test prompt".to_string()],
            system_prompt: "Test system prompt".to_string(),
            eval_api_endpoint: "https://api.openai.com/v1".to_string(),
            eval_env_var_api_key: "TEST_EVAL_API_KEY".to_string(),
            eval_model: "gpt-4".to_string(),
            eval_rate_limit_rps: 5.0,
            eval_prompt: "Evaluate for {categories}".to_string(),
            eval_system_prompt: "Test system prompt".to_string(),
            eval_categories: vec!["correctness".to_string(), "completeness".to_string()],
            storage_path: None,
        }
    }

    #[test]
    fn test_evaluator_new() {
        let evaluator = Evaluator::new();
        assert!(evaluator.last_api_request.is_none());
        assert!(evaluator.last_eval_request.is_none());
    }

    #[tokio::test]
    async fn test_enforce_rate_limit_no_limit() {
        let mut last_request = None;
        let start = TokioInstant::now();
        
        Evaluator::enforce_rate_limit(&mut last_request, 0.0).await;
        
        let elapsed = start.elapsed();
        assert!(elapsed < Duration::from_millis(10)); // Should return immediately
    }

    #[tokio::test]
    async fn test_enforce_rate_limit_negative_limit() {
        let mut last_request = None;
        let start = TokioInstant::now();
        
        Evaluator::enforce_rate_limit(&mut last_request, -1.0).await;
        
        let elapsed = start.elapsed();
        assert!(elapsed < Duration::from_millis(10)); // Should return immediately
    }

    #[tokio::test]
    async fn test_enforce_rate_limit_first_request() {
        let mut last_request = None;
        let start = TokioInstant::now();
        
        Evaluator::enforce_rate_limit(&mut last_request, 10.0).await;
        
        let elapsed = start.elapsed();
        assert!(elapsed < Duration::from_millis(10)); // Should not sleep on first request
        assert!(last_request.is_some());
    }

    #[tokio::test]
    async fn test_enforce_rate_limit_with_sleep() {
        let mut last_request = Some(Instant::now());
        let start = TokioInstant::now();
        
        // Set a low rate limit to force sleep
        Evaluator::enforce_rate_limit(&mut last_request, 100.0).await;
        
        let elapsed = start.elapsed();
        assert!(elapsed >= Duration::from_millis(8)); // Should have slept at least ~10ms minus some tolerance
    }

    #[tokio::test]
    async fn test_generate_response_missing_env_var() {
        let mut evaluator = Evaluator::new();
        let config = create_test_config();
        
        // Remove the environment variable if it exists
        unsafe {
            std::env::remove_var(&config.env_var_api_key);
        }
        
        let result = evaluator.generate_response(&config, "test prompt").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[tokio::test]
    async fn test_evaluate_response_missing_env_var() {
        let mut evaluator = Evaluator::new();
        let config = create_test_config();
        
        // Remove the evaluation environment variable if it exists
        unsafe {
            std::env::remove_var(&config.eval_env_var_api_key);
        }
        
        let result = evaluator.evaluate_response(&config, "prompt", "response").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[test]
    fn test_parse_evaluation_response_valid_json() {
        let evaluator = Evaluator::new();
        let response = r#"{"scores": {"correctness": 0.8, "completeness": 0.9}, "feedback": "Good response"}"#;
        let categories = vec!["correctness".to_string(), "completeness".to_string()];

        let result = evaluator
            .parse_evaluation_response(response, &categories)
            .unwrap();
        assert_eq!(result.scores.get("correctness"), Some(&0.8));
        assert_eq!(result.scores.get("completeness"), Some(&0.9));
        assert_eq!(result.feedback, "Good response");
        assert_eq!(result.raw_response, response);
    }

    #[test]
    fn test_parse_evaluation_response_embedded_json() {
        let evaluator = Evaluator::new();
        let response = r#"Here is the evaluation: {"scores": {"correctness": 0.7}, "feedback": "Decent"} That's all."#;
        let categories = vec!["correctness".to_string()];

        let result = evaluator
            .parse_evaluation_response(response, &categories)
            .unwrap();
        assert_eq!(result.scores.get("correctness"), Some(&0.7));
        assert_eq!(result.feedback, "Decent");
    }

    #[test]
    fn test_parse_evaluation_response_no_scores_object() {
        let evaluator = Evaluator::new();
        let response = r#"{"feedback": "No scores provided"}"#;
        let categories = vec!["correctness".to_string(), "completeness".to_string()];

        let result = evaluator
            .parse_evaluation_response(response, &categories)
            .unwrap();
        assert!(result.scores.is_empty());
        assert_eq!(result.feedback, "No scores provided");
    }

    #[test]
    fn test_parse_evaluation_response_missing_category_score() {
        let evaluator = Evaluator::new();
        let response = r#"{"scores": {"correctness": 0.8}, "feedback": "Missing completeness"}"#;
        let categories = vec!["correctness".to_string(), "completeness".to_string()];

        let result = evaluator
            .parse_evaluation_response(response, &categories)
            .unwrap();
        assert_eq!(result.scores.get("correctness"), Some(&0.8));
        assert_eq!(result.scores.get("completeness"), Some(&0.0)); // Default value
    }

    #[test]
    fn test_parse_evaluation_response_non_numeric_score() {
        let evaluator = Evaluator::new();
        let response = r#"{"scores": {"correctness": "invalid"}, "feedback": "Non-numeric score"}"#;
        let categories = vec!["correctness".to_string()];

        let result = evaluator
            .parse_evaluation_response(response, &categories)
            .unwrap();
        assert_eq!(result.scores.get("correctness"), Some(&0.0)); // Default for invalid
    }

    #[test]
    fn test_parse_evaluation_response_missing_feedback() {
        let evaluator = Evaluator::new();
        let response = r#"{"scores": {"correctness": 0.5}}"#;
        let categories = vec!["correctness".to_string()];

        let result = evaluator
            .parse_evaluation_response(response, &categories)
            .unwrap();
        assert_eq!(result.scores.get("correctness"), Some(&0.5));
        assert_eq!(result.feedback, "No feedback provided");
    }

    #[test]
    fn test_parse_evaluation_response_score_clamping() {
        let evaluator = Evaluator::new();
        let response = r#"{"scores": {"correctness": 1.5, "completeness": -0.5}, "feedback": "Clamping test"}"#;
        let categories = vec!["correctness".to_string(), "completeness".to_string()];

        let result = evaluator
            .parse_evaluation_response(response, &categories)
            .unwrap();
        assert_eq!(result.scores.get("correctness"), Some(&1.0)); // Clamped to 1.0
        assert_eq!(result.scores.get("completeness"), Some(&0.0)); // Clamped to 0.0
    }

    #[test]
    fn test_parse_evaluation_response_invalid_json() {
        let evaluator = Evaluator::new();
        let response = r#"invalid json content"#;
        let categories = vec!["correctness".to_string()];

        let result = evaluator.parse_evaluation_response(response, &categories);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_evaluation_response_no_closing_brace() {
        let evaluator = Evaluator::new();
        let response = r#"{"scores": {"correctness": 0.8"#; // Missing closing brace
        let categories = vec!["correctness".to_string()];

        let result = evaluator.parse_evaluation_response(response, &categories);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_evaluation_response_no_opening_brace() {
        let evaluator = Evaluator::new();
        let response = r#"scores": {"correctness": 0.8}}"#; // Missing opening brace
        let categories = vec!["correctness".to_string()];

        let result = evaluator.parse_evaluation_response(response, &categories);
        assert!(result.is_err());
    }

    #[test]
    fn test_calculate_statistics_normal_case() {
        let evaluator = Evaluator::new();
        let mut results = vec![];

        for scores in vec![
            vec![("correctness", 0.8), ("completeness", 0.9)],
            vec![("correctness", 0.6), ("completeness", 0.7)],
            vec![("correctness", 0.8), ("completeness", 0.8)],
        ] {
            let mut score_map = HashMap::new();
            for (category, score) in scores {
                score_map.insert(category.to_string(), score);
            }
            results.push(EvaluationResult {
                scores: score_map,
                feedback: "Test feedback".to_string(),
                raw_response: "{}".to_string(),
            });
        }

        let result_refs: Vec<&EvaluationResult> = results.iter().collect();
        let categories = vec!["correctness".to_string(), "completeness".to_string()];
        let stats = evaluator.calculate_statistics(&result_refs, &categories);

        // Mean: (0.8 + 0.6 + 0.8) / 3 = 0.733...
        assert!((stats.mean.get("correctness").unwrap() - 0.7333333333333333).abs() < 0.0001);
        // Median: 0.8 (middle value when sorted: 0.6, 0.8, 0.8)
        assert_eq!(stats.median.get("correctness"), Some(&0.8));
    }

    #[test]
    fn test_calculate_statistics_empty_results() {
        let evaluator = Evaluator::new();
        let results: Vec<&EvaluationResult> = vec![];
        let categories = vec!["correctness".to_string()];
        
        let stats = evaluator.calculate_statistics(&results, &categories);
        assert_eq!(stats.mean.get("correctness"), Some(&0.0));
        assert_eq!(stats.median.get("correctness"), Some(&0.0));
        assert_eq!(stats.mode.get("correctness"), Some(&0.0));
    }

    #[test]
    fn test_calculate_statistics_single_result() {
        let evaluator = Evaluator::new();
        let mut score_map = HashMap::new();
        score_map.insert("correctness".to_string(), 0.75);
        
        let result = EvaluationResult {
            scores: score_map,
            feedback: "Single result".to_string(),
            raw_response: "{}".to_string(),
        };
        
        let result_refs: Vec<&EvaluationResult> = vec![&result];
        let categories = vec!["correctness".to_string()];
        let stats = evaluator.calculate_statistics(&result_refs, &categories);
        
        assert_eq!(stats.mean.get("correctness"), Some(&0.75));
        assert_eq!(stats.median.get("correctness"), Some(&0.75));
        assert_eq!(stats.mode.get("correctness"), Some(&0.8)); // Rounded to 1 decimal
    }

    #[test]
    fn test_calculate_statistics_even_number_results() {
        let evaluator = Evaluator::new();
        let mut results = vec![];
        
        for score in vec![0.6, 0.7, 0.8, 0.9] {
            let mut score_map = HashMap::new();
            score_map.insert("correctness".to_string(), score);
            results.push(EvaluationResult {
                scores: score_map,
                feedback: "Test".to_string(),
                raw_response: "{}".to_string(),
            });
        }
        
        let result_refs: Vec<&EvaluationResult> = results.iter().collect();
        let categories = vec!["correctness".to_string()];
        let stats = evaluator.calculate_statistics(&result_refs, &categories);
    
        // Mean: (0.6 + 0.7 + 0.8 + 0.9) / 4 = 0.75
        assert!((stats.mean.get("correctness").unwrap() - 0.75).abs() < 1e-6);
        // Median: (0.7 + 0.8) / 2 = 0.75
        assert!((stats.median.get("correctness").unwrap() - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_calculate_statistics_missing_score_data() {
        let evaluator = Evaluator::new();
        let mut score_map = HashMap::new();
        score_map.insert("correctness".to_string(), 0.8);
        // Missing "completeness" category
        
        let result = EvaluationResult {
            scores: score_map,
            feedback: "Missing category test".to_string(),
            raw_response: "{}".to_string(),
        };
        
        let result_refs: Vec<&EvaluationResult> = vec![&result];
        let categories = vec!["correctness".to_string(), "completeness".to_string()];
        let stats = evaluator.calculate_statistics(&result_refs, &categories);
        
        assert_eq!(stats.mean.get("correctness"), Some(&0.8));
        assert_eq!(stats.mean.get("completeness"), Some(&0.0)); // Default for missing
    }

    #[test]
    fn test_calculate_statistics_mode_calculation() {
        let evaluator = Evaluator::new();
        let mut results = vec![];
        
        // Create results where 0.8 appears most frequently
        for score in vec![0.75, 0.8, 0.8, 0.8, 0.9] {
            let mut score_map = HashMap::new();
            score_map.insert("correctness".to_string(), score);
            results.push(EvaluationResult {
                scores: score_map,
                feedback: "Mode test".to_string(),
                raw_response: "{}".to_string(),
            });
        }
        
        let result_refs: Vec<&EvaluationResult> = results.iter().collect();
        let categories = vec!["correctness".to_string()];
        let stats = evaluator.calculate_statistics(&result_refs, &categories);
        
        assert_eq!(stats.mode.get("correctness"), Some(&0.8));
    }
}