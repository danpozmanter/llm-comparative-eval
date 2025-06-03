use crate::config::EvaluationConfig;
use crate::models::{EvaluationResult, ModelResponse, Statistics};
use anyhow::{Context, Result};
use async_openai::{Client, types::CreateChatCompletionRequestArgs};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::time::sleep;

/// Service for evaluating AI model responses with rate limiting
pub struct EvaluationService {
    /// Last request time for main API endpoint
    last_api_request: Option<Instant>,
    /// Last request time for evaluation API endpoint
    last_eval_request: Option<Instant>,
}

impl EvaluationService {
    /// Create a new evaluation service
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
        // Enforce rate limiting
        Self::enforce_rate_limit(&mut self.last_api_request, config.rate_limit_rps).await;

        let api_key = std::env::var(&config.env_var_api_key).with_context(|| {
            format!("Environment variable {} not found", config.env_var_api_key)
        })?;

        let openai_config = async_openai::config::OpenAIConfig::new()
            .with_api_key(api_key)
            .with_api_base(&config.api_endpoint);

        let client = Client::with_config(openai_config);

        let request = CreateChatCompletionRequestArgs::default()
            .model(&config.model)
            .messages([
                async_openai::types::ChatCompletionRequestSystemMessageArgs::default()
                    .content("You are a helpful assistant.")
                    .build()?
                    .into(),
                async_openai::types::ChatCompletionRequestUserMessageArgs::default()
                    .content(prompt.to_string())
                    .build()?
                    .into(),
            ])
            .temperature(config.temperature as f32)
            .max_tokens(config.max_tokens as u16)
            .build()?;

        let response = client
            .chat()
            .create(request)
            .await
            .context("Failed to generate response")?;

        let content = response
            .choices
            .first()
            .and_then(|choice| choice.message.content.as_ref())
            .unwrap_or(&String::new())
            .clone();

        let mut metadata = HashMap::new();
        if let Some(usage) = response.usage {
            metadata.insert("prompt_tokens".to_string(), json!(usage.prompt_tokens));
            metadata.insert(
                "completion_tokens".to_string(),
                json!(usage.completion_tokens),
            );
            metadata.insert("total_tokens".to_string(), json!(usage.total_tokens));
        }

        Ok(ModelResponse { content, metadata })
    }

    /// Evaluate a response using the evaluation model
    pub async fn evaluate_response(
        &mut self,
        config: &EvaluationConfig,
        prompt: &str,
        response: &str,
    ) -> Result<EvaluationResult> {
        // Enforce rate limiting for evaluation API
        Self::enforce_rate_limit(&mut self.last_eval_request, config.eval_rate_limit_rps).await;

        let eval_api_key = std::env::var(&config.eval_env_var_api_key).with_context(|| {
            format!(
                "Environment variable {} not found",
                config.eval_env_var_api_key
            )
        })?;

        let categories_str = config.eval_categories.join(", ");
        let eval_prompt = config.eval_prompt.replace("{categories}", &categories_str);

        let full_eval_prompt = format!(
            "{}\n\nOriginal Prompt: {}\nResponse to Evaluate: {}\n\nPlease provide scores (0.0 to 1.0) for each category and detailed feedback. Return as JSON with 'scores' and 'feedback' fields.",
            eval_prompt, prompt, response
        );

        let openai_config = async_openai::config::OpenAIConfig::new()
            .with_api_key(eval_api_key)
            .with_api_base(&config.eval_api_endpoint);

        let client = Client::with_config(openai_config);

        let request = CreateChatCompletionRequestArgs::default()
            .model(&config.eval_model)
            .messages([
                async_openai::types::ChatCompletionRequestSystemMessageArgs::default()
                    .content("You are an expert evaluator. Always return valid JSON.")
                    .build()?
                    .into(),
                async_openai::types::ChatCompletionRequestUserMessageArgs::default()
                    .content(full_eval_prompt)
                    .build()?
                    .into(),
            ])
            .temperature(0.1)
            .build()?;

        let eval_response = client
            .chat()
            .create(request)
            .await
            .context("Failed to generate evaluation response")?;

        let eval_content = eval_response
            .choices
            .first()
            .and_then(|choice| choice.message.content.as_ref())
            .unwrap_or(&String::new())
            .clone();

        self.parse_evaluation_response(&eval_content, &config.eval_categories)
    }

    /// Parse the evaluation response JSON
    fn parse_evaluation_response(
        &self,
        response: &str,
        categories: &[String],
    ) -> Result<EvaluationResult> {
        let parsed: Value = serde_json::from_str(response)
            .or_else(|_| {
                // Try to extract JSON from response if it's embedded in text
                if let Some(start) = response.find('{') {
                    if let Some(end) = response.rfind('}') {
                        serde_json::from_str(&response[start..=end])
                    } else {
                        Err(serde_json::from_str::<Value>("").unwrap_err())
                    }
                } else {
                    Err(serde_json::from_str::<Value>("").unwrap_err())
                }
            })
            .context("Failed to parse evaluation response as JSON")?;

        let mut scores = HashMap::new();

        if let Some(scores_obj) = parsed.get("scores").and_then(|s| s.as_object()) {
            for category in categories {
                if let Some(score) = scores_obj.get(category).and_then(|s| s.as_f64()) {
                    scores.insert(category.clone(), score.clamp(0.0, 1.0));
                } else {
                    scores.insert(category.clone(), 0.0);
                }
            }
        }

        let feedback = parsed
            .get("feedback")
            .and_then(|f| f.as_str())
            .unwrap_or("No feedback provided")
            .to_string();

        Ok(EvaluationResult {
            scores,
            feedback,
            raw_response: response.to_string(),
        })
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
            let mut scores: Vec<f64> = results
                .iter()
                .filter_map(|r| r.scores.get(category))
                .copied()
                .collect();

            if scores.is_empty() {
                mean.insert(category.clone(), 0.0);
                median.insert(category.clone(), 0.0);
                mode.insert(category.clone(), 0.0);
                continue;
            }

            // Calculate mean
            let sum: f64 = scores.iter().sum();
            mean.insert(category.clone(), sum / scores.len() as f64);

            // Calculate median
            scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mid = scores.len() / 2;
            let median_val = if scores.len() % 2 == 0 {
                (scores[mid - 1] + scores[mid]) / 2.0
            } else {
                scores[mid]
            };
            median.insert(category.clone(), median_val);

            // Calculate mode (most frequent value, rounded to 1 decimal place)
            let mut frequency = HashMap::new();
            for &score in &scores {
                let rounded = ((score * 10.0).round() as i32) as f64 / 10.0;
                *frequency.entry(rounded.to_bits()).or_insert(0) += 1;
            }
            let mode_val = frequency
                .iter()
                .max_by_key(|&(_, count)| count)
                .map(|(&bits, _)| f64::from_bits(bits))
                .unwrap_or(0.0);
            mode.insert(category.clone(), mode_val);
        }

        Statistics { mean, median, mode }
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
            api_endpoint: "https://api.openai.com/v1".to_string(),
            env_var_api_key: "TEST_API_KEY".to_string(),
            model: "gpt-4".to_string(),
            temperature: 0.7,
            max_tokens: 1000,
            rate_limit_rps: 10.0,
            prompts: vec!["Test prompt".to_string()],
            eval_api_endpoint: "https://api.openai.com/v1".to_string(),
            eval_env_var_api_key: "TEST_EVAL_API_KEY".to_string(),
            eval_model: "gpt-4".to_string(),
            eval_rate_limit_rps: 5.0,
            eval_prompt: "Evaluate for {categories}".to_string(),
            eval_categories: vec!["correctness".to_string(), "completeness".to_string()],
            storage_path: None,
        }
    }

    #[test]
    fn test_evaluation_service_new() {
        let service = EvaluationService::new();
        assert!(service.last_api_request.is_none());
        assert!(service.last_eval_request.is_none());
    }

    #[tokio::test]
    async fn test_enforce_rate_limit_no_limit() {
        let mut last_request = None;
        let start = TokioInstant::now();
        
        EvaluationService::enforce_rate_limit(&mut last_request, 0.0).await;
        
        let elapsed = start.elapsed();
        assert!(elapsed < Duration::from_millis(10)); // Should return immediately
    }

    #[tokio::test]
    async fn test_enforce_rate_limit_negative_limit() {
        let mut last_request = None;
        let start = TokioInstant::now();
        
        EvaluationService::enforce_rate_limit(&mut last_request, -1.0).await;
        
        let elapsed = start.elapsed();
        assert!(elapsed < Duration::from_millis(10)); // Should return immediately
    }

    #[tokio::test]
    async fn test_enforce_rate_limit_first_request() {
        let mut last_request = None;
        let start = TokioInstant::now();
        
        EvaluationService::enforce_rate_limit(&mut last_request, 10.0).await;
        
        let elapsed = start.elapsed();
        assert!(elapsed < Duration::from_millis(10)); // Should not sleep on first request
        assert!(last_request.is_some());
    }

    #[tokio::test]
    async fn test_enforce_rate_limit_with_sleep() {
        let mut last_request = Some(Instant::now());
        let start = TokioInstant::now();
        
        // Set a low rate limit to force sleep
        EvaluationService::enforce_rate_limit(&mut last_request, 100.0).await;
        
        let elapsed = start.elapsed();
        assert!(elapsed >= Duration::from_millis(8)); // Should have slept at least ~10ms minus some tolerance
    }

    #[tokio::test]
    async fn test_generate_response_missing_env_var() {
        let mut service = EvaluationService::new();
        let config = create_test_config();
        
        // Remove the environment variable if it exists
        unsafe {
            std::env::remove_var(&config.env_var_api_key);
        }
        
        let result = service.generate_response(&config, "test prompt").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[tokio::test]
    async fn test_evaluate_response_missing_env_var() {
        let mut service = EvaluationService::new();
        let config = create_test_config();
        
        // Remove the evaluation environment variable if it exists
        unsafe {
            std::env::remove_var(&config.eval_env_var_api_key);
        }
        
        let result = service.evaluate_response(&config, "prompt", "response").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[test]
    fn test_parse_evaluation_response_valid_json() {
        let service = EvaluationService::new();
        let response = r#"{"scores": {"correctness": 0.8, "completeness": 0.9}, "feedback": "Good response"}"#;
        let categories = vec!["correctness".to_string(), "completeness".to_string()];

        let result = service
            .parse_evaluation_response(response, &categories)
            .unwrap();
        assert_eq!(result.scores.get("correctness"), Some(&0.8));
        assert_eq!(result.scores.get("completeness"), Some(&0.9));
        assert_eq!(result.feedback, "Good response");
        assert_eq!(result.raw_response, response);
    }

    #[test]
    fn test_parse_evaluation_response_embedded_json() {
        let service = EvaluationService::new();
        let response = r#"Here is the evaluation: {"scores": {"correctness": 0.7}, "feedback": "Decent"} That's all."#;
        let categories = vec!["correctness".to_string()];

        let result = service
            .parse_evaluation_response(response, &categories)
            .unwrap();
        assert_eq!(result.scores.get("correctness"), Some(&0.7));
        assert_eq!(result.feedback, "Decent");
    }

    #[test]
    fn test_parse_evaluation_response_no_scores_object() {
        let service = EvaluationService::new();
        let response = r#"{"feedback": "No scores provided"}"#;
        let categories = vec!["correctness".to_string(), "completeness".to_string()];

        let result = service
            .parse_evaluation_response(response, &categories)
            .unwrap();
        assert!(result.scores.is_empty());
        assert_eq!(result.feedback, "No scores provided");
    }

    #[test]
    fn test_parse_evaluation_response_missing_category_score() {
        let service = EvaluationService::new();
        let response = r#"{"scores": {"correctness": 0.8}, "feedback": "Missing completeness"}"#;
        let categories = vec!["correctness".to_string(), "completeness".to_string()];

        let result = service
            .parse_evaluation_response(response, &categories)
            .unwrap();
        assert_eq!(result.scores.get("correctness"), Some(&0.8));
        assert_eq!(result.scores.get("completeness"), Some(&0.0)); // Default value
    }

    #[test]
    fn test_parse_evaluation_response_non_numeric_score() {
        let service = EvaluationService::new();
        let response = r#"{"scores": {"correctness": "invalid"}, "feedback": "Non-numeric score"}"#;
        let categories = vec!["correctness".to_string()];

        let result = service
            .parse_evaluation_response(response, &categories)
            .unwrap();
        assert_eq!(result.scores.get("correctness"), Some(&0.0)); // Default for invalid
    }

    #[test]
    fn test_parse_evaluation_response_missing_feedback() {
        let service = EvaluationService::new();
        let response = r#"{"scores": {"correctness": 0.5}}"#;
        let categories = vec!["correctness".to_string()];

        let result = service
            .parse_evaluation_response(response, &categories)
            .unwrap();
        assert_eq!(result.scores.get("correctness"), Some(&0.5));
        assert_eq!(result.feedback, "No feedback provided");
    }

    #[test]
    fn test_parse_evaluation_response_score_clamping() {
        let service = EvaluationService::new();
        let response = r#"{"scores": {"correctness": 1.5, "completeness": -0.5}, "feedback": "Clamping test"}"#;
        let categories = vec!["correctness".to_string(), "completeness".to_string()];

        let result = service
            .parse_evaluation_response(response, &categories)
            .unwrap();
        assert_eq!(result.scores.get("correctness"), Some(&1.0)); // Clamped to 1.0
        assert_eq!(result.scores.get("completeness"), Some(&0.0)); // Clamped to 0.0
    }

    #[test]
    fn test_parse_evaluation_response_invalid_json() {
        let service = EvaluationService::new();
        let response = r#"invalid json content"#;
        let categories = vec!["correctness".to_string()];

        let result = service.parse_evaluation_response(response, &categories);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_evaluation_response_no_closing_brace() {
        let service = EvaluationService::new();
        let response = r#"{"scores": {"correctness": 0.8"#; // Missing closing brace
        let categories = vec!["correctness".to_string()];

        let result = service.parse_evaluation_response(response, &categories);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_evaluation_response_no_opening_brace() {
        let service = EvaluationService::new();
        let response = r#"scores": {"correctness": 0.8}}"#; // Missing opening brace
        let categories = vec!["correctness".to_string()];

        let result = service.parse_evaluation_response(response, &categories);
        assert!(result.is_err());
    }

    #[test]
    fn test_calculate_statistics_normal_case() {
        let service = EvaluationService::new();
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
        let stats = service.calculate_statistics(&result_refs, &categories);

        // Mean: (0.8 + 0.6 + 0.8) / 3 = 0.733...
        assert!((stats.mean.get("correctness").unwrap() - 0.7333333333333333).abs() < 0.0001);
        // Median: 0.8 (middle value when sorted: 0.6, 0.8, 0.8)
        assert_eq!(stats.median.get("correctness"), Some(&0.8));
    }

    #[test]
    fn test_calculate_statistics_empty_results() {
        let service = EvaluationService::new();
        let results: Vec<&EvaluationResult> = vec![];
        let categories = vec!["correctness".to_string()];
        
        let stats = service.calculate_statistics(&results, &categories);
        assert_eq!(stats.mean.get("correctness"), Some(&0.0));
        assert_eq!(stats.median.get("correctness"), Some(&0.0));
        assert_eq!(stats.mode.get("correctness"), Some(&0.0));
    }

    #[test]
    fn test_calculate_statistics_single_result() {
        let service = EvaluationService::new();
        let mut score_map = HashMap::new();
        score_map.insert("correctness".to_string(), 0.75);
        
        let result = EvaluationResult {
            scores: score_map,
            feedback: "Single result".to_string(),
            raw_response: "{}".to_string(),
        };
        
        let result_refs: Vec<&EvaluationResult> = vec![&result];
        let categories = vec!["correctness".to_string()];
        let stats = service.calculate_statistics(&result_refs, &categories);
        
        assert_eq!(stats.mean.get("correctness"), Some(&0.75));
        assert_eq!(stats.median.get("correctness"), Some(&0.75));
        assert_eq!(stats.mode.get("correctness"), Some(&0.8)); // Rounded to 1 decimal
    }

    #[test]
    fn test_calculate_statistics_even_number_results() {
        let service = EvaluationService::new();
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
        let stats = service.calculate_statistics(&result_refs, &categories);
    
        // Mean: (0.6 + 0.7 + 0.8 + 0.9) / 4 = 0.75
        assert!((stats.mean.get("correctness").unwrap() - 0.75).abs() < 1e-6);
        // Median: (0.7 + 0.8) / 2 = 0.75
        assert!((stats.median.get("correctness").unwrap() - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_calculate_statistics_missing_score_data() {
        let service = EvaluationService::new();
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
        let stats = service.calculate_statistics(&result_refs, &categories);
        
        assert_eq!(stats.mean.get("correctness"), Some(&0.8));
        assert_eq!(stats.mean.get("completeness"), Some(&0.0)); // Default for missing
    }

    #[test]
    fn test_calculate_statistics_mode_calculation() {
        let service = EvaluationService::new();
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
        let stats = service.calculate_statistics(&result_refs, &categories);
        
        assert_eq!(stats.mode.get("correctness"), Some(&0.8));
    }
}