use clap::Parser;
use std::path::PathBuf;

mod config;
mod evaluation;
mod models;
mod output;
mod runner;

use crate::config::Config;
use crate::output::OutputFormat;
use crate::runner::Runner;

/// AI Model Evaluation CLI - Run prompts against AI models and evaluate responses
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the TOML configuration file
    run_file: PathBuf,

    /// Output format: plain or json
    #[arg(short, long, default_value = "plain")]
    output: OutputFormat,

    /// Verbose output - show progress for each API request
    #[arg(short, long)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let config = Config::from_file(&args.run_file)?;
    let mut runner = Runner::new(config, args.verbose);

    let results = runner.run_evaluations().await?;

    output::print_results(&results, args.output);

    Ok(())
}
