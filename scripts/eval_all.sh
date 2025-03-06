while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -p|--predictions_path)
            predictions_path="$2"
            shift 2
            ;;
        -o|--out_dir)
            out_dir="$2"
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

echo ${predictions_path}
echo ${out_dir}

datasets=("ECB" "SciCo" "MultiNews" "MusiQue" "OpenASP" "FuseReviews")

for ds in "${datasets[@]}"; do
  echo "Evaluating ${ds}"
  python scripts/evaluate_dataset.py --ds_name "${ds}" --predictions_dir "${predictions_path}/${ds}" --output_path "${out_dir}/${ds}/results.json"
  python scripts/evaluation/statistical_analysis.py --results_path "${out_dir}/${ds}/results.json"
done

models=("Mistral-7B" "Llama3-8B" "Llama3-70B" "Mixtral-8x7B" "Mixtral-8x22B" "Gemma1.1-2B" "Gemma1.1-7B")

for model in "${models[@]}"; do
  echo "Evaluating ${model}"
  python scripts/evaluation/sample_length_function.py --results_dir "${out_dir}" --model_name "${model}"
done

python scripts/evaluation/compare_models.py --results_dir "${out_dir}"