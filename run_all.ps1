$models = @("LogisticRegression","RandomForest","MLP", "XGBoost", "SVM")
$datasets = @("Obesity", "Heart_disease", "Breast_cancer", "Diabetes")
$grouping_methods = @("ungrouped", "kplus", "bicriterion", "group_dissimilar", "random")
# $datasets = @("Diabetes")
# $grouping_methods = @("random", "kplus")
 
foreach ($model in $models) {
  foreach ($dataset in $datasets) {
    foreach ($grouping in $grouping_methods) {
      Write-Host "=================================================="
      Write-Host "Running with: $model | $dataset | $grouping | SHAP=$shap_mode"
      Write-Host "=================================================="
      py main.py --model $model --dataset $dataset --grouping_method $grouping
    }
  }
}
