
# Dataset Description

This dataset for the Kaggle competition (both train and test sets) was generated from a deep learning model trained on the Loan Approval Prediction dataset.

## Files

- **trainloan.csv** - The training dataset, where `loan_status` is the binary target.
- **testloan.csv** - The test dataset, where your objective is to predict the probability of the target `loan_status` for each row.
- **sample_submission.csv** - A sample submission file in the correct format.

## Evaluation

Submissions are evaluated using the area under the ROC curve (AUC) based on the predicted probabilities and the ground truth targets.

## Submission File

For each `id` row in the test set, you must predict the target `loan_status`. The file should contain a header and follow this format:

```plaintext
id,loan_status
58645,0.5
58646,0.5
58647,0.5
...
```

## Timeline

- Start Date: October 1, 2024
- Entry Deadline: October 31, 2024
- Team Merger Deadline: October 31, 2024
- Final Submission Deadline: October 31, 2024

## Citation

Walter Reade and Ashley Chow. Loan Approval Prediction. https://kaggle.com/competitions/playground-series-s4e10, 2024. Kaggle.
```
