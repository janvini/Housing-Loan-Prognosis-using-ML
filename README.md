# Loan Prediction Model 

## Project Overview

The Loan Prediction Model is a machine learning project designed to predict the likelihood of loan approval based on various features related to loan applicants. The project involves building, evaluating, and fine-tuning different machine learning models to identify the most accurate predictor of loan approval. 

## Dataset

### File Structure

- `Loan_dataset.csv`: The dataset used for training and testing the model.
- `Smartmodel.ipynb`: The Jupyter Notebook containing the code for data analysis, pre-processing, and model training.

### Dataset Details

The dataset contains 614 entries with 13 columns:

- `Loan_ID`: Unique Loan Identifier (Dropped during processing)
- `Gender`: Male/Female
- `Married`: Applicant's marital status
- `Dependents`: Number of dependents
- `Education`: Applicant's education level (Graduate/Not Graduate)
- `Self_Employed`: Whether the applicant is self-employed
- `ApplicantIncome`: Income of the applicant
- `CoapplicantIncome`: Income of the co-applicant
- `LoanAmount`: Loan amount in thousands
- `Loan_Amount_Term`: Term of the loan in months
- `Credit_History`: Credit history (1.0 - Good, 0.0 - Bad)
- `Property_Area`: Area where the property is located (Urban/Semiurban/Rural)
- `Loan_Status`: Loan approval status (Y/N)

## Data Pre-Processing

### Handling Missing Values

- **LoanAmount**: Filled missing values with the mean of the column.
- **Loan_Amount_Term**: Filled missing values with the mean of the column.
- **Credit_History**: Filled missing values with the mean of the column.
- **Gender, Married, Dependents, Self_Employed**: Filled missing values with the mode of their respective columns.

### Feature Engineering

- **Dropped `Loan_ID`**: This column was dropped as it does not contribute to the prediction.

## Exploratory Data Analysis (EDA)

### Uni-variate Analysis

- **Distributions**: Analyzed the distribution of individual features using `distplot`, `histplot`, and `countplot` from Seaborn.
- **Pie Chart**: Visualized the distribution of `Property_Area`.

### Bivariate Analysis

- **Count Plots**: Visualized relationships between categorical variables like `Married` and `Gender`, `Education` and `Self_Employed`.
- **Scatter Plots**: Analyzed relationships between numerical features like `ApplicantIncome`, `CoapplicantIncome`, and `LoanAmount`.

### Multivariate Analysis

- **Heatmap**: Visualized correlations between numerical features.
- **Line Plot**: Analyzed the relationship between multiple numerical features.

## Model Building

### Data Splitting

The dataset was split into training and testing sets using `train_test_split`.

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=['Loan_Status'])  # Features
y = df['Loan_Status']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### Model Training

Multiple models were trained and compared, including:

- **Decision Tree Classifier**
- **Random Forest Classifier**
- **K-Nearest Neighbors (KNN)**
- **XGBoost Classifier**

Additionally, Logistic Regression was explored with hyperparameter tuning.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f'Training Accuracy: {train_accuracy}')
print(f'Testing Accuracy: {test_accuracy}')
```

### Data Rescaling

- **MaxAbsScaler**: Applied to scale features to ensure similar magnitudes.

### Handling Imbalance

- **RandomUnderSampler**: Used from the `imblearn` library to address class imbalance.

### Hyperparameter Tuning

Optimized the models using GridSearchCV.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga']
}

grid = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print(f'Best Parameters: {grid.best_params_}')
print(f'Best Training Accuracy: {grid.best_score_}')
```

### Model Evaluation

Evaluated models using:

- **Accuracy Score**
- **Classification Report**
- **Confusion Matrix**
- **F1 Score**

The models were assessed based on precision, recall, and F1-score for both classes.

### Final Model Evaluation and Saving

```python
best_model = grid.best_estimator_
y_test_pred = best_model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Final Test Accuracy: {test_accuracy}')

import joblib
joblib.dump(best_model, 'loan_prediction_model.pkl')
```

## Results

The performance of each model was compared, and the best-performing model was selected based on evaluation metrics such as accuracy, precision, recall, and F1-score.

## Conclusion

This project successfully developed a machine learning model to predict loan approval. The dataset was meticulously pre-processed, and various models were trained and evaluated. The best model, optimized through hyperparameter tuning, achieved satisfactory performance metrics.

## Dependencies

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Seaborn
- Matplotlib
- Imbalanced-learn (`imblearn`)
- XGBoost

Install the required libraries using:

```bash
pip install numpy pandas scikit-learn seaborn matplotlib imbalanced-learn xgboost
```

## Running the Project

1. Clone the repository.
2. Install the required dependencies.
3. Run the `Smartmodel.ipynb` notebook to execute the data analysis, model training, and evaluation steps.

## Model Deployment

To deploy the model, you can use web frameworks such as Flask, FastAPI, or Django to create an application that utilizes the trained model.
