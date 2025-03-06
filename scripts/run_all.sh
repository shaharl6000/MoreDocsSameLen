while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -c|--config)
            config="$2"
            shift 2
            ;;
      *)
            echo "Error: Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

export PYTHONPATH=./

#models=("meta-llama/Meta-Llama-3-8B-Instruct" "mistralai/Mistral-7B-Instruct-v0.2" "google/gemma-1.1-2b-it" "mistralai/Mixtral-8x7B-Instruct-v0.1" "google/gemma-1.1-7b-it" "meta-llama/Meta-Llama-3-70B-Instruct" "mistralai/Mixtral-8x22B-Instruct-v0.1")
models=("meta-llama/Meta-Llama-3-8B-Instruct" "mistralai/Mistral-7B-Instruct-v0.2" "google/gemma-1.1-2b-it" "mistralai/Mixtral-8x7B-Instruct-v0.1" "google/gemma-1.1-7b-it" "meta-llama/Meta-Llama-3-70B-Instruct" "mistralai/Mixtral-8x22B-Instruct-v0.1")

for model in "${models[@]}"; do
    python scripts/run_model_predictions.py --model_name "${model}" --config "${config}"
done