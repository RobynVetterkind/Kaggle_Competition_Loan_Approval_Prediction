#Import Libraries

import pandas as pd                                                                    #library for data manipulation and analysis, primarily used for handling data in tabular form (like DataFrames).
import numpy as np                                                                     #library for numerical computations in Python, used for operations on arrays and matrices.
from sklearn.preprocessing import StandardScaler, OneHotEncoder                        #StandardScaler: Standardizes features by removing the mean and scaling to unit variance.OneHotEncoder: Converts categorical variables into a form that could be provided to ML algorithms to do a better job in prediction.
from sklearn.compose import ColumnTransformer                                          #Allows you to apply different preprocessing steps to different columns of your data.
from sklearn.pipeline import Pipeline                                                  #Chains together multiple processing steps in a single object.
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier        #Ensemble methods for classification that improve predictive accuracy by combining the predictions of multiple base estimators.
from sklearn.model_selection import train_test_split, GridSearchCV                     #train_test_split: Splits arrays or matrices into random train and test subsets. GridSearchCV: searches through a specified hyperparameter space to find the best model.
from sklearn.metrics import roc_auc_score                                              #Computes the area under the Receiver Operating Characteristic curve (AUC), a performance measure for classification problems.
from sklearn.impute import SimpleImputer                                               #Used to handle missing values by filling them with a specified strategy (mean, median, constant, etc.).

#Load Data
#Loads the training and test datasets from CSV files into DataFrames.

train = pd.read_csv(r"C:\Users\robyn\Downloads\trainloan.csv")
test = pd.read_csv(r"C:\Users\robyn\Downloads\testloan.csv")

#Prepare Features and Target
#X: Feature set obtained by dropping the loan_status (target variable) and id (non-predictive identifier) columns from the training data.
#y: Target variable (loan_status) which I want to predict.

X = train.drop(['loan_status', 'id'], axis=1)
y = train['loan_status']

#Identify Numeric and Categorical Features
#numeric_features: Columns in X that are of numeric data types (integers and floats).
#categorical_features: Columns in X that are of object data types (usually strings representing categories).

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

#Preprocessing Pipeline

#Numeric Features: 
#SimpleImputer(strategy='median'): Fills missing values in numeric features with the median of that feature.
#StandardScaler(): Scales numeric features to have a mean of 0 and a standard deviation of 1.

#Categorical Features:
#SimpleImputer(strategy='constant', fill_value='missing'): Fills missing values in categorical features with the string "missing".
#OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'): Converts categorical variables into a one-hot encoded format, dropping the first category to avoid multicollinearity.

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ]), categorical_features)
    ])


#Model Pipeline
#Combines the preprocessing steps and the classifier (in this case, a Gradient Boosting Classifier) into a single pipeline.

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])


#Split Data into Training and Validation Sets
#Splits the data into training (80%) and validation (20%) sets.

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


#Hyperparameter Tuning

#param_grid: Defines the hyperparameters to be tuned for the GradientBoostingClassifier.

param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [3, 4, 5],
    'classifier__learning_rate': [0.01, 0.1, 0.2]
}


#GridSearchCV: Performs a grid search on the specified parameters using cross-validation (5-fold in this case) to find the best combination of parameters based on AUC scoring. n_jobs=-1 allows it to use all available processors.
#grid_search.fit(X_train, y_train): Trains the model on the training data.

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)


#Best Model Selection
#Retrieves the best model found during the grid search.

best_model = grid_search.best_estimator_


#Model Evaluation

#predict_proba(X_val)[:, 1]: Gets the predicted probabilities of the positive class (loan approval) for the validation set.
#roc_auc_score(y_val, val_predictions): Calculates the AUC score for the validation predictions.

val_predictions = best_model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_predictions)
print(f"Validation AUC: {val_auc}")


#Prepare Test Data and Generate Predictions

#X_test: Prepares the test data by dropping the id column.
#test_predictions: Predicts probabilities for the test set.

X_test = test.drop('id', axis=1)
test_predictions = best_model.predict_proba(X_test)[:, 1]


#submission: Creates a DataFrame for submission with IDs and predicted loan statuses.
#Saves the submission DataFrame to a CSV file (improved_submission.csv) without row indices and confirms successful creation.

submission = pd.DataFrame({
    'id': test['id'],
    'loan_status': test_predictions
})
submission.to_csv('improved_submission.csv', index=False)
print("Improved submission file created successfully.")
