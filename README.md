Dataset Description:

The dataset for this Kaggle competition (both train and test) was generated from a deep learning model trained on the Loan Approval Prediction dataset.

Files: 

train.csv - the training dataset; loan_status is the binary target.
test.csv - the test dataset; your objective is to predict probability of the target loan_status for each row.
sample_submission.csv - a sample submission file in the correct format.

Evaluation:

Submissions are evaluated using area under the ROC curve using the predicted probabilities and the ground truth targets.

Submission File:

For each id row in the test set, you must predict target loan_status. The file should contain a header and have the following format:

id,loan_status

58645,0.5 |
58646,0.5 |
58647,0.5 |
etc.


Timeline:

Start Date - October 1, 2024
Entry Deadline - Same as the Final Submission Deadline
Team Merger Deadline - Same as the Final Submission Deadline
Final Submission Deadline - October 31, 2024


Citation:
Walter Reade and Ashley Chow. Loan Approval Prediction. https://kaggle.com/competitions/playground-series-s4e10, 2024. Kaggle.
