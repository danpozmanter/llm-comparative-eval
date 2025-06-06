use crate::config::{Config, EvaluationConfig};
use crate::evaluation::Evaluator;
use crate::models::{EvaluationResult, FinalResults, PromptResult};
use anyhow::{Context, Result};
use std::path::Path;

/// Main runner that orchestrates the evaluation process
pub struct Runner {
    config: Config,
    evaluator: Evaluator,
    verbose: bool,
}

impl Runner {
    /// Create a new runner with the given configuration
    pub fn new(config: Config, verbose: bool) -> Self {
        Self {
            config,
            evaluator: Evaluator::new(),
            verbose,
        }
    }

    /// Run all evaluations defined in the configuration
    pub async fn run_evaluations(&mut self) -> Result<Vec<FinalResults>> {
        let mut all_results = Vec::new();
        let total_evaluations = self.config.evaluations.len();
        let evaluations = self.config.evaluations.clone();

        for (eval_index, eval_config) in evaluations.iter().enumerate() {
            let eval_num = eval_index + 1;
            let results = self
                .run_single_evaluation(eval_config, eval_num, total_evaluations)
                .await?;
            all_results.push(results);
        }

        Ok(all_results)
    }

    /// Run evaluation for a single configuration
    async fn run_single_evaluation(
        &mut self,
        config: &EvaluationConfig,
        eval_num: usize,
        total_evaluations: usize,
    ) -> Result<FinalResults> {
        if self.verbose {
            println!("Running evaluation {:?}", config.title);
        }
        let prompt_results = self.process_all_prompts(config, eval_num, total_evaluations).await?;
        let statistics = self.calculate_evaluation_statistics(&prompt_results, config, eval_num, total_evaluations);
        let final_results = FinalResults { statistics, results: prompt_results };
        
        self.store_results_if_configured(&final_results, config, eval_num, total_evaluations)?;
        
        Ok(final_results)
    }

    /// Process all prompts for a single evaluation
    async fn process_all_prompts(
        &mut self,
        config: &EvaluationConfig,
        eval_num: usize,
        total_evaluations: usize,
    ) -> Result<Vec<PromptResult>> {
        let mut prompt_results = Vec::new();
        let total_prompts = config.prompts.len();

        for (prompt_index, prompt) in config.prompts.iter().enumerate() {
            let prompt_num = prompt_index + 1;
            
            self.log_prompt_processing(prompt_num, total_prompts, eval_num, total_evaluations);
            
            let result = self
                .evaluate_single_prompt(config, prompt, prompt_num, total_prompts, eval_num, total_evaluations)
                .await
                .with_context(|| format!("Failed to evaluate prompt: {}", prompt))?;
            
            prompt_results.push(result);
        }

        Ok(prompt_results)
    }

    /// Log prompt processing status if verbose mode is enabled
    fn log_prompt_processing(
        &self,
        prompt_num: usize,
        total_prompts: usize,
        eval_num: usize,
        total_evaluations: usize,
    ) {
        if self.verbose {
            println!(
                "Processing prompt {}/{}, evaluation {}/{}",
                prompt_num, total_prompts, eval_num, total_evaluations
            );
        }
    }

    /// Calculate statistics for the evaluation
    fn calculate_evaluation_statistics(
        &mut self,
        prompt_results: &[PromptResult],
        config: &EvaluationConfig,
        eval_num: usize,
        total_evaluations: usize,
    ) -> crate::models::Statistics {
        if self.verbose {
            println!(
                "Calculating statistics for evaluation {}/{}",
                eval_num, total_evaluations
            );
        }

        let evaluations: Vec<&EvaluationResult> = prompt_results.iter().map(|r| &r.evaluation).collect();
        self.evaluator.calculate_statistics(&evaluations, &config.eval_categories)
    }

    /// Store results if storage path is configured
    fn store_results_if_configured(
        &self,
        final_results: &FinalResults,
        config: &EvaluationConfig,
        eval_num: usize,
        total_evaluations: usize,
    ) -> Result<()> {
        if let Some(storage_path) = &config.storage_path {
            self.log_storage_operation(eval_num, total_evaluations, storage_path);
            self.store_results(final_results, storage_path)?;
        }
        Ok(())
    }

    /// Log storage operation if verbose mode is enabled
    fn log_storage_operation(&self, eval_num: usize, total_evaluations: usize, storage_path: &str) {
        if self.verbose {
            println!(
                "Storing results for evaluation {}/{} to {}",
                eval_num, total_evaluations, storage_path
            );
        }
    }

    /// Evaluate a single prompt and return the complete result
    async fn evaluate_single_prompt(
        &mut self,
        config: &EvaluationConfig,
        prompt: &str,
        prompt_num: usize,
        total_prompts: usize,
        eval_num: usize,
        total_evaluations: usize,
    ) -> Result<PromptResult> {
        let response = self.generate_prompt_response(config, prompt, prompt_num, total_prompts, eval_num, total_evaluations).await?;
        let evaluation = self.evaluate_prompt_response(config, prompt, &response.content, prompt_num, total_prompts, eval_num, total_evaluations).await?;

        Ok(PromptResult {
            prompt: prompt.to_string(),
            response,
            evaluation,
        })
    }

    /// Generate response for a single prompt
    async fn generate_prompt_response(
        &mut self,
        config: &EvaluationConfig,
        prompt: &str,
        prompt_num: usize,
        total_prompts: usize,
        eval_num: usize,
        total_evaluations: usize,
    ) -> Result<crate::models::ModelResponse> {
        self.log_response_generation(prompt_num, total_prompts, eval_num, total_evaluations);
        
        self.evaluator
            .generate_response(config, prompt)
            .await
            .context("Failed to generate response")
    }

    /// Log response generation if verbose mode is enabled
    fn log_response_generation(
        &self,
        prompt_num: usize,
        total_prompts: usize,
        eval_num: usize,
        total_evaluations: usize,
    ) {
        if self.verbose {
            println!(
                "  → Generating response for prompt {}/{}, evaluation {}/{}",
                prompt_num, total_prompts, eval_num, total_evaluations
            );
        }
    }

    /// Evaluate a single prompt response
    async fn evaluate_prompt_response(
        &mut self,
        config: &EvaluationConfig,
        prompt: &str,
        response_content: &str,
        prompt_num: usize,
        total_prompts: usize,
        eval_num: usize,
        total_evaluations: usize,
    ) -> Result<EvaluationResult> {
        self.log_response_evaluation(prompt_num, total_prompts, eval_num, total_evaluations);
        
        self.evaluator
            .evaluate_response(config, prompt, response_content)
            .await
            .context("Failed to evaluate response")
    }

    /// Log response evaluation if verbose mode is enabled
    fn log_response_evaluation(
        &self,
        prompt_num: usize,
        total_prompts: usize,
        eval_num: usize,
        total_evaluations: usize,
    ) {
        if self.verbose {
            println!(
                "  → Evaluating response for prompt {}/{}, evaluation {}/{}",
                prompt_num, total_prompts, eval_num, total_evaluations
            );
        }
    }

    /// Store results to a JSON file
    fn store_results(&self, final_results: &FinalResults, path: &str) -> Result<()> {
        let json_content = self.serialize_results(final_results)?;
        self.ensure_directory_exists(path)?;
        self.write_results_file(path, &json_content)?;
        self.log_storage_success(path);
        
        Ok(())
    }

    /// Serialize results to JSON
    fn serialize_results(&self, final_results: &FinalResults) -> Result<String> {
        serde_json::to_string_pretty(final_results)
            .context("Failed to serialize results to JSON")
    }

    /// Ensure the directory for the results file exists
    fn ensure_directory_exists(&self, path: &str) -> Result<()> {
        if let Some(parent) = Path::new(path).parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create directory: {}", parent.display()))?;
        }
        Ok(())
    }

    /// Write results to file
    fn write_results_file(&self, path: &str, content: &str) -> Result<()> {
        std::fs::write(path, content)
            .with_context(|| format!("Failed to write results to: {}", path))
    }

    /// Log successful storage operation
    fn log_storage_success(&self, path: &str) {
        println!("Results stored to: {}", path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EvaluationConfig;
    use tempfile::tempdir;

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
            eval_rate_limit_rps: 10.0,
            eval_prompt: "Evaluate for {categories}".to_string(),
            eval_system_prompt: "Test system prompt".to_string(),
            eval_categories: vec!["correctness".to_string()],
            storage_path: None,
        }
    }

    #[test]
    fn test_store_results() {
        use crate::models::Statistics;
        use std::collections::HashMap;

        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_results.json");

        let config = Config {
            evaluations: vec![create_test_config()],
        };
        let runner = Runner::new(config, false);

        let final_results = FinalResults {
            statistics: Statistics {
                mean: HashMap::new(),
                median: HashMap::new(),
                mode: HashMap::new(),
            },
            results: vec![],
        };

        runner
            .store_results(&final_results, file_path.to_str().unwrap())
            .unwrap();

        assert!(file_path.exists());
        let content = std::fs::read_to_string(&file_path).unwrap();
        // Should contain both statistics and results
        assert!(content.contains("statistics"));
        assert!(content.contains("results"));
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::config::EvaluationConfig;
        use crate::models::{EvaluationResult, ModelResponse, Statistics};
        use tempfile::tempdir;
        use std::collections::HashMap;
    
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
                eval_rate_limit_rps: 10.0,
                eval_prompt: "Evaluate for {categories}".to_string(),
                eval_system_prompt: "Test system prompt".to_string(),
                eval_categories: vec!["correctness".to_string()],
                storage_path: None,
            }
        }
    
        fn create_config_with_multiple_prompts() -> EvaluationConfig {
            let mut config = create_test_config();
            config.prompts = vec![
                "First prompt".to_string(),
                "Second prompt".to_string(),
                "Third prompt".to_string(),
            ];
            config.eval_categories = vec![
                "correctness".to_string(),
                "completeness".to_string(),
                "clarity".to_string(),
            ];
            config
        }
    
        // Mock Evaluator for testing
        struct MockEvaluator {
            should_fail_generate: bool,
            should_fail_evaluate: bool,
        }
    
        impl MockEvaluator {
            fn new() -> Self {
                Self {
                    should_fail_generate: false,
                    should_fail_evaluate: false,
                }
            }
    
            fn with_generate_failure() -> Self {
                Self {
                    should_fail_generate: true,
                    should_fail_evaluate: false,
                }
            }
    
            fn with_evaluate_failure() -> Self {
                Self {
                    should_fail_generate: false,
                    should_fail_evaluate: true,
                }
            }
    
            async fn generate_response(
                &mut self,
                _config: &EvaluationConfig,
                prompt: &str,
            ) -> Result<ModelResponse> {
                if self.should_fail_generate {
                    return Err(anyhow::anyhow!("Mock generate failure"));
                }
    
                let mut metadata = HashMap::new();
                metadata.insert("prompt_tokens".to_string(), serde_json::json!(10));
                metadata.insert("completion_tokens".to_string(), serde_json::json!(50));
                metadata.insert("total_tokens".to_string(), serde_json::json!(60));
    
                Ok(ModelResponse {
                    content: format!("Mock response for: {}", prompt),
                    metadata,
                })
            }
    
            async fn evaluate_response(
                &mut self,
                _config: &EvaluationConfig,
                _prompt: &str,
                _response: &str,
            ) -> Result<EvaluationResult> {
                if self.should_fail_evaluate {
                    return Err(anyhow::anyhow!("Mock evaluate failure"));
                }
    
                let mut scores = HashMap::new();
                scores.insert("correctness".to_string(), 0.85);
                scores.insert("completeness".to_string(), 0.90);
    
                Ok(EvaluationResult {
                    scores,
                    feedback: "Mock evaluation feedback".to_string(),
                    raw_response: r#"{"scores": {"correctness": 0.85}, "feedback": "Good"}"#.to_string(),
                })
            }
    
            fn calculate_statistics(
                &self,
                results: &[&EvaluationResult],
                categories: &[String],
            ) -> Statistics {
                let mut mean = HashMap::new();
                let mut median = HashMap::new();
                let mut mode = HashMap::new();
    
                for category in categories {
                    let scores: Vec<f64> = results
                        .iter()
                        .filter_map(|r| r.scores.get(category))
                        .copied()
                        .collect();
    
                    if !scores.is_empty() {
                        let sum: f64 = scores.iter().sum();
                        mean.insert(category.clone(), sum / scores.len() as f64);
                        median.insert(category.clone(), scores[0]);
                        mode.insert(category.clone(), scores[0]);
                    } else {
                        mean.insert(category.clone(), 0.0);
                        median.insert(category.clone(), 0.0);
                        mode.insert(category.clone(), 0.0);
                    }
                }
    
                Statistics { mean, median, mode }
            }
        }
    
        // Test Runner struct that allows injecting mock service
        struct TestRunner {
            config: Config,
            mock_service: MockEvaluator,
            verbose: bool,
        }
    
        impl TestRunner {
            fn new(config: Config, verbose: bool) -> Self {
                Self {
                    config,
                    mock_service: MockEvaluator::new(),
                    verbose,
                }
            }
    
            fn with_generate_failure(config: Config, verbose: bool) -> Self {
                Self {
                    config,
                    mock_service: MockEvaluator::with_generate_failure(),
                    verbose,
                }
            }
    
            fn with_evaluate_failure(config: Config, verbose: bool) -> Self {
                Self {
                    config,
                    mock_service: MockEvaluator::with_evaluate_failure(),
                    verbose,
                }
            }
    
            async fn run_evaluations(&mut self) -> Result<Vec<FinalResults>> {
                let mut all_results = Vec::new();
                let total_evaluations = self.config.evaluations.len();
    
                let evaluations = self.config.evaluations.clone();
    
                for (eval_index, eval_config) in evaluations.iter().enumerate() {
                    let eval_num = eval_index + 1;
                    let results = self
                        .run_single_evaluation(&eval_config, eval_num, total_evaluations)
                        .await?;
                    all_results.push(results);
                }
    
                Ok(all_results)
            }
    
            async fn run_single_evaluation(
                &mut self,
                config: &EvaluationConfig,
                eval_num: usize,
                total_evaluations: usize,
            ) -> Result<FinalResults> {
                let mut prompt_results = Vec::new();
                let total_prompts = config.prompts.len();
    
                for (prompt_index, prompt) in config.prompts.iter().enumerate() {
                    let prompt_num = prompt_index + 1;
    
                    if self.verbose {
                        println!(
                            "Processing prompt {}/{}, evaluation {}/{}",
                            prompt_num, total_prompts, eval_num, total_evaluations
                        );
                    }
    
                    let result = self
                        .evaluate_single_prompt(
                            config,
                            prompt,
                            prompt_num,
                            total_prompts,
                            eval_num,
                            total_evaluations,
                        )
                        .await
                        .with_context(|| format!("Failed to evaluate prompt: {}", prompt))?;
                    prompt_results.push(result);
                }
    
                if self.verbose {
                    println!(
                        "Calculating statistics for evaluation {}/{}",
                        eval_num, total_evaluations
                    );
                }
    
                let evaluations: Vec<&EvaluationResult> =
                    prompt_results.iter().map(|r| &r.evaluation).collect();
                let statistics = self
                    .mock_service
                    .calculate_statistics(&evaluations, &config.eval_categories);
    
                let final_results = FinalResults {
                    statistics,
                    results: prompt_results,
                };
    
                if let Some(storage_path) = &config.storage_path {
                    if self.verbose {
                        println!(
                            "Storing results for evaluation {}/{} to {}",
                            eval_num, total_evaluations, storage_path
                        );
                    }
                    self.store_results(&final_results, storage_path)?;
                }
    
                Ok(final_results)
            }
    
            async fn evaluate_single_prompt(
                &mut self,
                config: &EvaluationConfig,
                prompt: &str,
                prompt_num: usize,
                total_prompts: usize,
                eval_num: usize,
                total_evaluations: usize,
            ) -> Result<PromptResult> {
                if self.verbose {
                    println!(
                        "  → Generating response for prompt {}/{}, evaluation {}/{}",
                        prompt_num, total_prompts, eval_num, total_evaluations
                    );
                }
    
                let response = self
                    .mock_service
                    .generate_response(config, prompt)
                    .await
                    .context("Failed to generate response")?;
    
                if self.verbose {
                    println!(
                        "  → Evaluating response for prompt {}/{}, evaluation {}/{}",
                        prompt_num, total_prompts, eval_num, total_evaluations
                    );
                }
    
                let evaluation = self
                    .mock_service
                    .evaluate_response(config, prompt, &response.content)
                    .await
                    .context("Failed to evaluate response")?;
    
                Ok(PromptResult {
                    prompt: prompt.to_string(),
                    response,
                    evaluation,
                })
            }
    
            fn store_results(&self, final_results: &FinalResults, path: &str) -> Result<()> {
                let json_content = serde_json::to_string_pretty(final_results)
                    .context("Failed to serialize results to JSON")?;
    
                if let Some(parent) = Path::new(path).parent() {
                    std::fs::create_dir_all(parent)
                        .with_context(|| format!("Failed to create directory: {}", parent.display()))?;
                }
    
                std::fs::write(path, json_content)
                    .with_context(|| format!("Failed to write results to: {}", path))?;
    
                if self.verbose {
                    println!("Results stored to: {}", path);
                } else {
                    println!("Results stored to: {}", path);
                }
                Ok(())
            }
        }
    
        #[test]
        fn test_runner_new() {
            let config = Config {
                evaluations: vec![create_test_config()],
            };
            let runner = Runner::new(config.clone(), false);
            assert!(!runner.verbose);
            
            let runner_verbose = Runner::new(config, true);
            assert!(runner_verbose.verbose);
        }
    
        #[tokio::test]
        async fn test_run_evaluations_single_config() {
            let config = Config {
                evaluations: vec![create_test_config()],
            };
            let mut runner = TestRunner::new(config, false);
            
            let results = runner.run_evaluations().await.unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].results.len(), 1);
        }
    
        #[tokio::test]
        async fn test_run_evaluations_multiple_configs() {
            let config = Config {
                evaluations: vec![
                    create_test_config(),
                    create_config_with_multiple_prompts(),
                ],
            };
            let mut runner = TestRunner::new(config, true);
            
            let results = runner.run_evaluations().await.unwrap();
            assert_eq!(results.len(), 2);
            assert_eq!(results[0].results.len(), 1); // First config has 1 prompt
            assert_eq!(results[1].results.len(), 3); // Second config has 3 prompts
        }
    
        #[tokio::test]
        async fn test_run_evaluations_with_storage() {
            let temp_dir = tempdir().unwrap();
            let storage_path = temp_dir.path().join("test_results.json");
            
            let mut config = create_test_config();
            config.storage_path = Some(storage_path.to_string_lossy().to_string());
            
            let full_config = Config {
                evaluations: vec![config],
            };
            let mut runner = TestRunner::new(full_config, true);
            
            let results = runner.run_evaluations().await.unwrap();
            assert_eq!(results.len(), 1);
            assert!(storage_path.exists());
        }
    
        #[tokio::test]
        async fn test_run_evaluations_generate_failure() {
            let config = Config {
                evaluations: vec![create_test_config()],
            };
            let mut runner = TestRunner::with_generate_failure(config, false);
            
            let result = runner.run_evaluations().await;
            assert!(result.is_err());
            assert!(result.unwrap_err().to_string().contains("Failed to evaluate prompt"));
        }
    
        #[tokio::test]
        async fn test_run_evaluations_evaluate_failure() {
            let config = Config {
                evaluations: vec![create_test_config()],
            };
            let mut runner = TestRunner::with_evaluate_failure(config, false);
            
            let result = runner.run_evaluations().await;
            assert!(result.is_err());
            assert!(result.unwrap_err().to_string().contains("Failed to evaluate prompt"));
        }
    
        #[tokio::test]
        async fn test_run_single_evaluation_verbose() {
            let config = create_config_with_multiple_prompts();
            let mut runner = TestRunner::new(Config { evaluations: vec![] }, true);
            
            let result = runner.run_single_evaluation(&config, 1, 2).await.unwrap();
            assert_eq!(result.results.len(), 3);
            assert!(result.statistics.mean.contains_key("correctness"));
        }
    
        #[tokio::test]
        async fn test_run_single_evaluation_non_verbose() {
            let config = create_test_config();
            let mut runner = TestRunner::new(Config { evaluations: vec![] }, false);
            
            let result = runner.run_single_evaluation(&config, 1, 1).await.unwrap();
            assert_eq!(result.results.len(), 1);
        }
    
        #[tokio::test]
        async fn test_evaluate_single_prompt_verbose() {
            let config = create_test_config();
            let mut runner = TestRunner::new(Config { evaluations: vec![] }, true);
            
            let result = runner
                .evaluate_single_prompt(&config, "Test prompt", 1, 2, 1, 3)
                .await
                .unwrap();
            
            assert_eq!(result.prompt, "Test prompt");
            assert!(result.response.content.contains("Mock response"));
            assert_eq!(result.evaluation.feedback, "Mock evaluation feedback");
        }
    
        #[tokio::test]
        async fn test_evaluate_single_prompt_non_verbose() {
            let config = create_test_config();
            let mut runner = TestRunner::new(Config { evaluations: vec![] }, false);
            
            let result = runner
                .evaluate_single_prompt(&config, "Test prompt", 1, 1, 1, 1)
                .await
                .unwrap();
            
            assert_eq!(result.prompt, "Test prompt");
        }
    
        #[test]
        fn test_store_results() {
            let temp_dir = tempdir().unwrap();
            let file_path = temp_dir.path().join("test_results.json");
    
            let config = Config {
                evaluations: vec![create_test_config()],
            };
            let runner = Runner::new(config, false);
    
            let mut mean = HashMap::new();
            mean.insert("correctness".to_string(), 0.85);
    
            let final_results = FinalResults {
                statistics: Statistics {
                    mean,
                    median: HashMap::new(),
                    mode: HashMap::new(),
                },
                results: vec![],
            };
    
            runner
                .store_results(&final_results, file_path.to_str().unwrap())
                .unwrap();
    
            assert!(file_path.exists());
            let content = std::fs::read_to_string(&file_path).unwrap();
            assert!(content.contains("statistics"));
            assert!(content.contains("results"));
            assert!(content.contains("correctness"));
        }
    
        #[test]
        fn test_store_results_verbose() {
            let temp_dir = tempdir().unwrap();
            let file_path = temp_dir.path().join("verbose_results.json");
    
            let config = Config {
                evaluations: vec![create_test_config()],
            };
            let runner = Runner::new(config, true);
    
            let final_results = FinalResults {
                statistics: Statistics {
                    mean: HashMap::new(),
                    median: HashMap::new(),
                    mode: HashMap::new(),
                },
                results: vec![],
            };
    
            runner
                .store_results(&final_results, file_path.to_str().unwrap())
                .unwrap();
    
            assert!(file_path.exists());
        }
    
        #[test]
        fn test_store_results_with_nested_directory() {
            let temp_dir = tempdir().unwrap();
            let nested_path = temp_dir.path().join("nested").join("directory").join("results.json");
    
            let config = Config {
                evaluations: vec![create_test_config()],
            };
            let runner = Runner::new(config, false);
    
            let final_results = FinalResults {
                statistics: Statistics {
                    mean: HashMap::new(),
                    median: HashMap::new(),
                    mode: HashMap::new(),
                },
                results: vec![],
            };
    
            runner
                .store_results(&final_results, nested_path.to_str().unwrap())
                .unwrap();
    
            assert!(nested_path.exists());
        }
    
        #[test]
        fn test_store_results_with_complex_data() {
            let temp_dir = tempdir().unwrap();
            let file_path = temp_dir.path().join("complex_results.json");
    
            let config = Config {
                evaluations: vec![create_test_config()],
            };
            let runner = Runner::new(config, false);
    
            let mut scores = HashMap::new();
            scores.insert("correctness".to_string(), 0.95);
            scores.insert("completeness".to_string(), 0.88);
    
            let evaluation_result = EvaluationResult {
                scores: scores.clone(),
                feedback: "Comprehensive feedback with detailed analysis".to_string(),
                raw_response: r#"{"scores": {"correctness": 0.95}, "feedback": "Great!"}"#.to_string(),
            };
    
            let mut metadata = HashMap::new();
            metadata.insert("prompt_tokens".to_string(), serde_json::json!(42));
            metadata.insert("completion_tokens".to_string(), serde_json::json!(150));
    
            let model_response = ModelResponse {
                content: "Detailed model response content".to_string(),
                metadata,
            };
    
            let prompt_result = PromptResult {
                prompt: "Complex test prompt".to_string(),
                response: model_response,
                evaluation: evaluation_result,
            };
    
            let mut mean = HashMap::new();
            mean.insert("correctness".to_string(), 0.95);
            mean.insert("completeness".to_string(), 0.88);
    
            let statistics = Statistics {
                mean,
                median: HashMap::new(),
                mode: HashMap::new(),
            };
    
            let final_results = FinalResults {
                statistics,
                results: vec![prompt_result],
            };
    
            runner
                .store_results(&final_results, file_path.to_str().unwrap())
                .unwrap();
    
            assert!(file_path.exists());
            let content = std::fs::read_to_string(&file_path).unwrap();
            assert!(content.contains("Complex test prompt"));
            assert!(content.contains("Detailed model response content"));
            assert!(content.contains("Comprehensive feedback"));
            assert!(content.contains("0.95"));
        }
    
        #[test]
        fn test_store_results_serialization_failure() {
            
            // Create a path that will cause a serialization error by using invalid characters
            let invalid_path = "/dev/null/invalid_path_that_cannot_exist";
            
            let config = Config {
                evaluations: vec![create_test_config()],
            };
            let runner = Runner::new(config, false);
    
            let final_results = FinalResults {
                statistics: Statistics {
                    mean: HashMap::new(),
                    median: HashMap::new(),
                    mode: HashMap::new(),
                },  
                results: vec![],
            };
    
            let result = runner.store_results(&final_results, invalid_path);
            assert!(result.is_err());
        }
    
        #[test]
        fn test_multiple_evaluation_configs() {
            let config1 = create_test_config();
            let mut config2 = create_test_config();
            config2.model = "gpt-3.5-turbo".to_string();
            config2.prompts = vec!["Different prompt".to_string()];
    
            let config = Config {
                evaluations: vec![config1, config2],
            };
    
            let runner = Runner::new(config, true);
            assert_eq!(runner.config.evaluations.len(), 2);
            assert!(runner.verbose);
        }
    
        #[test]
        fn test_config_with_storage_path() {
            let temp_dir = tempdir().unwrap();
            let storage_path = temp_dir.path().join("auto_storage.json");
            
            let mut config = create_test_config();
            config.storage_path = Some(storage_path.to_string_lossy().to_string());
    
            let full_config = Config {
                evaluations: vec![config],
            };
    
            let runner = Runner::new(full_config, false);
            assert!(runner.config.evaluations[0].storage_path.is_some());
        }
    
        #[test]
        fn test_evaluation_categories() {
            let mut config = create_test_config();
            config.eval_categories = vec![
                "accuracy".to_string(),
                "relevance".to_string(),
                "coherence".to_string(),
                "completeness".to_string(),
            ];
    
            let full_config = Config {
                evaluations: vec![config],
            };
    
            let runner = Runner::new(full_config, false);
            assert_eq!(runner.config.evaluations[0].eval_categories.len(), 4);
            assert!(runner.config.evaluations[0].eval_categories.contains(&"accuracy".to_string()));
            assert!(runner.config.evaluations[0].eval_categories.contains(&"relevance".to_string()));
        }
    
        #[tokio::test]
        async fn test_empty_evaluations_config() {
            let config = Config {
                evaluations: vec![],
            };
            let mut runner = TestRunner::new(config, false);
            
            let results = runner.run_evaluations().await.unwrap();
            assert_eq!(results.len(), 0);
        }
    
        #[tokio::test]
        async fn test_config_with_no_prompts() {
            let mut config = create_test_config();
            config.prompts = vec![];
            
            let full_config = Config {
                evaluations: vec![config],
            };
            let mut runner = TestRunner::new(full_config, false);
            
            let results = runner.run_evaluations().await.unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].results.len(), 0);
        }
    
        #[tokio::test]
        async fn test_comprehensive_workflow() {
            let temp_dir = tempdir().unwrap();
            
            // Create multiple configs with different settings
            let mut config1 = create_test_config();
            config1.prompts = vec!["First test".to_string(), "Second test".to_string()];
            config1.storage_path = Some(temp_dir.path().join("config1.json").to_string_lossy().to_string());
            
            let mut config2 = create_config_with_multiple_prompts();
            config2.storage_path = Some(temp_dir.path().join("config2.json").to_string_lossy().to_string());
            
            let full_config = Config {
                evaluations: vec![config1, config2],
            };
            
            let mut runner = TestRunner::new(full_config, true);
            let results = runner.run_evaluations().await.unwrap();
            
            // Verify results structure
            assert_eq!(results.len(), 2);
            assert_eq!(results[0].results.len(), 2); // First config: 2 prompts
            assert_eq!(results[1].results.len(), 3); // Second config: 3 prompts
            
            // Verify storage files were created
            assert!(temp_dir.path().join("config1.json").exists());
            assert!(temp_dir.path().join("config2.json").exists());
            
            // Verify statistics were calculated
            assert!(!results[0].statistics.mean.is_empty());
            assert!(!results[1].statistics.mean.is_empty());
        }
    }
}