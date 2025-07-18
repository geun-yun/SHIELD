#!/bin/bash

models=("LogisticRegression" "SVM" "RandomForest")
datasets=("Obesity" "Heart_disease" "Breat_cancer" "Diabetes")
grouping_methods=("ungrouped" "bicriterion" "group_dissimilar")
shap_modes=("each_label" "aggregated")

for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    for grouping in "${grouping_methods[@]}"; do
      for shap in "${shap_modes[@]}"; do
        echo "=================================================="
        echo "Running with: $model | $dataset | $grouping | $shap"
        echo "=================================================="
        python main.py \
          --model "$model" \
          --dataset "$dataset" \
          --grouping_method "$grouping" \
          --SHAP "$shap"
      done
    done
  done
done
