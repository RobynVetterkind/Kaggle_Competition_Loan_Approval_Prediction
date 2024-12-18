
# Dataset Description

This dataset for the Kaggle competition (both train and test sets) was generated from a deep learning model trained on the Loan Approval Prediction dataset.

# Importance

Every loan carries inherent financial risk. Predictive models allow organizations to mathematically model potential losses, understanding that a single percentage point of default reduction can save millions in potential unrecovered funds. This isn't just about preventing loss, but strategically allocating financial resources to borrowers most likely to generate sustainable returns, thereby creating a more intelligent, efficient lending system that balances institutional protection with economic opportunity.

Loan prediction plays a pivotal role in optimizing credit risk management and enhancing operational workflows within financial institutions. By employing advanced machine learning models, predictive analytics, and statistical techniques, data engineers ensure the robust collection, preprocessing, and transformation of large datasets, such as transaction histories, borrower profiles, and credit scores. Data analysts leverage these datasets to build and fine-tune predictive models that quantify the likelihood of loan default, utilizing algorithms like Gradient Boosting, XGBoost, or deep learning frameworks to extract complex patterns. Business analysts work closely with data scientists to interpret model outputs and integrate them into decision-support systems. The deployment of such models allows for dynamic risk assessments, minimizing default risks while automating loan approval workflows. These insights enable lenders to optimize their resource allocation, personalize loan offerings based on risk profiles, and mitigate exposure to bad debts. Furthermore, the continuous feedback loop from the model allows for real-time adjustments, enhancing both operational efficiency and profitability in a competitive financial landscape.


## Files

- **trainloan.csv** - The training dataset, where `loan_status` is the binary target.
- **testloan.csv** - The test dataset, where your objective is to predict the probability of the target `loan_status` for each row.
- **sample_submission.csv** - A sample submission file in the correct format.
- **improved_submission.csv** - My submission file containing Loan IDs' and approval status in decimal format.
- **Robyn_Vetterkind_fixedcodeforcomp(v2)_FINAL_SUBMISSION.py** - Containing Python code from V2 of submission three.

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

## Attempts


I implemented several enhancements in the code to improve validation AUC. Feature engineering was incorporated before splitting the dataset, adding derived features such as ratios and interaction terms while addressing high cardinality in categorical variables within the preprocessing pipeline. To handle class imbalance, SMOTE was applied after splitting the training data, and class_weight='balanced' was introduced in the GradientBoostingClassifier. Hyperparameter tuning was expanded with additional ranges, including parameters like max_depth, subsample, and learning_rate, and RandomizedSearchCV was used to reduce computational load. I also tested alternative models, including XGBoost, LightGBM, and CatBoost, by replacing the GradientBoostingClassifier in the pipeline. Despite these efforts, the program's execution became computationally intensive, with runtime extending for hours without producing results. The process was hindered by limited SSD storage on my personal device, which prevented successful completion and output generation. As a result, the final code submission does not include the desired enhancements and represents the last of three submissions, reverting to the simpler version of the workflow.


## Timeline

- Start Date: October 1, 2024
- Entry Deadline: October 31, 2024
- Team Merger Deadline: October 31, 2024
- Final Submission Deadline: October 31, 2024

## Citation

Walter Reade and Ashley Chow. Loan Approval Prediction. https://kaggle.com/competitions/playground-series-s4e10, 2024. Kaggle.
```
