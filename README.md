# llm-comparative-eval
Compare how llm models stack up.

## Intro

Prompt engineering can be tricky, as can selecting the right model for the use case at hand.

I wrote this to have a resource friendly and reliable CLI tool to evaluate prompts and models.

1. For each evaluation, select the model and give it a list of prompts. (These can be quite different, or small variations).

2. Then select the model and prompt to use to evaluate the results.

3. Look at the results from prompt to prompt, or look at the overall stats for the entire evaluation.

## Usage

### Build

```bash
cargo build --release
```

### Run

```bash
llm-comparative-eval runfile.toml
```

or

```bash
llm-comparative-eval -v runfile.toml
```

## Run Files

Example:

```toml
[[evaluations]]
title = "General Knowledge"
api_endpoint = "https://api.together.xyz/v1"
env_var_api_key = "TOGETHER_AI_API_KEY"
model = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
temperature = 0.7
max_tokens = 500
rate_limit_rps = 0.25

prompts = [
    "What is the capital of France and why is it historically significant?",
    "Explain quantum entanglement in simple terms that a high school student could understand.",
    "What are the main causes of climate change and what can individuals do to help?",
    "Compare and contrast democracy and authoritarianism as forms of government.",
    "Explain the concept of compound interest and provide a practical example."
]

eval_api_endpoint = "https://api.together.xyz/v1"
eval_env_var_api_key = "TOGETHER_AI_API_KEY"
eval_model = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
eval_rate_limit_rps = 0.25
eval_prompt = "Evaluate this response for {categories}. Provide detailed feedback on accuracy, clarity, and completeness."
eval_categories = ["accuracy", "clarity", "completeness", "helpfulness"]
storage_path = "./results/general_knowledge.json"
```