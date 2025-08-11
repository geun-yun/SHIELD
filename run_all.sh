#!/bin/bash

models=("SVM")
datasets=("Obesity" "Heart_disease" "Breast_cancer" "Diabetes")
grouping_methods=("ungrouped" "bicriterion" "group_dissimilar" "random")

for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    for grouping in "${grouping_methods[@]}"; do
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
