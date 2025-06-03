# llm-comparative-eval
Compare how llm models stack up

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