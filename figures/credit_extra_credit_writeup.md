# Credit Approval Extra Credit Write-up

## EC1: Additional algorithm
Added Gaussian Naive Bayes as a third standalone model in addition to
Decision Tree and KNN. All algorithms were evaluated with stratified
10-fold CV using accuracy and F1-score.

## EC3: Ensemble
Built a heterogeneous ensemble composed of:
- Decision Tree
- KNN
- Gaussian Naive Bayes
- Random Forest

For each CV fold, every model is trained on its own bootstrap sample
drawn from the training fold. Final predictions are produced by
majority voting.

## Results table
| Model | Accuracy | F1-score |
|---|---:|---:|
| Decision Tree (thr=0.6) | 0.8638 | 0.8623 |
| KNN (k=3) | 0.8133 | 0.7848 |
| GaussianNB (vs=1e-07) | 0.6723 | 0.4643 |
| Ensemble (DT+KNN+GNB+RF) | 0.8298 | 0.7901 |

## Brief interpretation
- Ensemble voting improves robustness by averaging model-specific errors.
- If ensemble accuracy/F1 exceeds individual baselines, it indicates
  complementary decision boundaries across model families.
- If one standalone model remains best, this suggests low disagreement
  among models or insufficient diversity in the base learners.
