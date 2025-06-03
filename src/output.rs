use crate::models::FinalResults;
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

/// Output format options
#[derive(Debug, Clone, ValueEnum, Serialize, Deserialize)]
pub enum OutputFormat {
    Plain,
    Json,
}

/// Print evaluation results in the specified format
pub fn print_results(results: &[FinalResults], format: OutputFormat) {
    match format {
        OutputFormat::Plain => print_plain(results),
        OutputFormat::Json => print_json(results),
    }
}

/// Print results in plain text format
fn print_plain(results: &[FinalResults]) {
    for (i, result) in results.iter().enumerate() {
        println!("=== Evaluation {} ===", i + 1);
        println!();

        // Print statistics
        println!("📊 STATISTICS");
        println!("-------------");
        print_statistics_plain(&result.statistics);
        println!();

        // Print individual results
        println!("📝 DETAILED RESULTS");
        println!("-------------------");
        for (j, prompt_result) in result.results.iter().enumerate() {
            println!("Result #{}", j + 1);
            println!("Prompt: {}", prompt_result.prompt);
            println!("Response: {}", prompt_result.response.content);
            println!("Evaluation Scores:");
            for (category, score) in &prompt_result.evaluation.scores {
                println!("  • {}: {:.3}", category, score);
            }
            println!("Feedback: {}", prompt_result.evaluation.feedback);
            println!();
        }

        if i < results.len() - 1 {
            println!("{}", "=".repeat(50));
            println!();
        }
    }
}

/// Print statistics in plain text format
fn print_statistics_plain(stats: &crate::models::Statistics) {
    let categories: Vec<_> = stats.mean.keys().collect();

    if categories.is_empty() {
        println!("No statistics available.");
        return;
    }

    // Print header
    println!(
        "{:<15} {:<8} {:<8} {:<8}",
        "Category", "Mean", "Median", "Mode"
    );
    println!("{}", "-".repeat(45));

    for category in categories {
        let mean = stats.mean.get(category).unwrap_or(&0.0);
        let median = stats.median.get(category).unwrap_or(&0.0);
        let mode = stats.mode.get(category).unwrap_or(&0.0);

        println!(
            "{:<15} {:<8.3} {:<8.3} {:<8.3}",
            category, mean, median, mode
        );
    }
}

/// Print results in JSON format
fn print_json(results: &[FinalResults]) {
    match serde_json::to_string_pretty(results) {
        Ok(json) => println!("{}", json),
        Err(e) => eprintln!("Error serializing results to JSON: {}", e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{EvaluationResult, ModelResponse, PromptResult, Statistics};
    use std::collections::HashMap;

    fn create_test_results() -> Vec<FinalResults> {
        let mut scores = HashMap::new();
        scores.insert("correctness".to_string(), 0.8);

        let mut mean = HashMap::new();
        mean.insert("correctness".to_string(), 0.8);

        let evaluation = EvaluationResult {
            scores,
            feedback: "Test feedback".to_string(),
            raw_response: "{}".to_string(),
        };

        let response = ModelResponse {
            content: "Test response".to_string(),
            metadata: HashMap::new(),
        };

        let prompt_result = PromptResult {
            prompt: "Test prompt".to_string(),
            response,
            evaluation,
        };

        let statistics = Statistics {
            mean,
            median: HashMap::new(),
            mode: HashMap::new(),
        };

        vec![FinalResults {
            statistics,
            results: vec![prompt_result],
        }]
    }

    #[test]
    fn test_json_output() {
        let results = create_test_results();
        // This test mainly ensures the JSON serialization doesn't panic
        print_json(&results);
        // In a real test, you might capture stdout to verify the output
    }

    #[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{EvaluationResult, ModelResponse, PromptResult, Statistics};
    use std::collections::HashMap;

    fn create_test_results() -> Vec<FinalResults> {
        let mut scores = HashMap::new();
        scores.insert("correctness".to_string(), 0.8);
        scores.insert("completeness".to_string(), 0.9);

        let mut mean = HashMap::new();
        mean.insert("correctness".to_string(), 0.8);
        mean.insert("completeness".to_string(), 0.85);

        let mut median = HashMap::new();
        median.insert("correctness".to_string(), 0.75);
        median.insert("completeness".to_string(), 0.9);

        let mut mode = HashMap::new();
        mode.insert("correctness".to_string(), 0.8);
        mode.insert("completeness".to_string(), 0.9);

        let evaluation = EvaluationResult {
            scores: scores.clone(),
            feedback: "Test feedback with detailed analysis".to_string(),
            raw_response: r#"{"scores": {"correctness": 0.8, "completeness": 0.9}, "feedback": "Good"}"#.to_string(),
        };

        let mut metadata = HashMap::new();
        metadata.insert("tokens".to_string(), serde_json::json!(150));

        let response = ModelResponse {
            content: "This is a comprehensive test response that covers multiple aspects".to_string(),
            metadata,
        };

        let prompt_result = PromptResult {
            prompt: "What is artificial intelligence and how does it work?".to_string(),
            response,
            evaluation,
        };

        let statistics = Statistics {
            mean,
            median,
            mode,
        };

        vec![FinalResults {
            statistics,
            results: vec![prompt_result],
        }]
    }

    fn create_multiple_results() -> Vec<FinalResults> {
        let mut results = create_test_results();
        
        // Add a second evaluation result
        let mut scores2 = HashMap::new();
        scores2.insert("accuracy".to_string(), 0.95);
        
        let evaluation2 = EvaluationResult {
            scores: scores2,
            feedback: "Excellent accuracy".to_string(),
            raw_response: "{}".to_string(),
        };

        let response2 = ModelResponse {
            content: "Second response".to_string(),
            metadata: HashMap::new(),
        };

        let prompt_result2 = PromptResult {
            prompt: "Second prompt".to_string(),
            response: response2,
            evaluation: evaluation2,
        };

        let mut mean2 = HashMap::new();
        mean2.insert("accuracy".to_string(), 0.95);

        let statistics2 = Statistics {
            mean: mean2,
            median: HashMap::new(),
            mode: HashMap::new(),
        };

        results.push(FinalResults {
            statistics: statistics2,
            results: vec![prompt_result2],
        });

        results
    }

    #[test]
    fn test_json_output() {
        let results = create_test_results();
        // Capture stdout to verify JSON output
        print_json(&results);
        // This test ensures JSON serialization works without panicking
    }

    #[test]
    fn test_json_output_multiple_results() {
        let results = create_multiple_results();
        print_json(&results);
    }

    #[test]
    fn test_plain_output_single_result() {
        let results = create_test_results();
        print_plain(&results);
    }

    #[test]
    fn test_plain_output_multiple_results() {
        let results = create_multiple_results();
        print_plain(&results);
    }

    #[test]
    fn test_print_results_plain_format() {
        let results = create_test_results();
        print_results(&results, OutputFormat::Plain);
    }

    #[test]
    fn test_print_results_json_format() {
        let results = create_test_results();
        print_results(&results, OutputFormat::Json);
    }

    #[test]
    fn test_print_statistics_plain_empty() {
        let empty_stats = Statistics {
            mean: HashMap::new(),
            median: HashMap::new(),
            mode: HashMap::new(),
        };
        print_statistics_plain(&empty_stats);
    }

    #[test]
    fn test_print_statistics_plain_with_data() {
        let mut mean = HashMap::new();
        mean.insert("correctness".to_string(), 0.123456789);
        mean.insert("completeness".to_string(), 0.987654321);

        let mut median = HashMap::new();
        median.insert("correctness".to_string(), 0.111111111);
        median.insert("completeness".to_string(), 0.999999999);

        let mut mode = HashMap::new();
        mode.insert("correctness".to_string(), 0.1);
        mode.insert("completeness".to_string(), 1.0);

        let stats = Statistics { mean, median, mode };
        print_statistics_plain(&stats);
    }

    #[test]
    fn test_json_serialization_error_handling() {
        // Test with empty results
        let empty_results: Vec<FinalResults> = vec![];
        print_json(&empty_results);
    }

    #[test]
    fn test_output_format_variants() {
        // Test that OutputFormat enum variants work
        let plain = OutputFormat::Plain;
        let json = OutputFormat::Json;
        
        let results = create_test_results();
        print_results(&results, plain);
        print_results(&results, json);
    }
}
}
