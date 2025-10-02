# Titanic Survival Prediction 🚢

This repository contains a Jupyter Notebook that walks through a complete machine learning workflow to predict the survival of passengers on the Titanic. The project uses the famous Titanic dataset from Kaggle, focusing on data preprocessing, feature engineering, and model training to achieve a reliable prediction model.

## 📋 Project Overview

The goal of this project is to build a machine learning model that predicts whether a passenger on the Titanic survived or not. The process involves several key stages:

1.  **Data Exploration & Visualization:** Understanding the dataset, identifying correlations between features, and visualizing the data to gain insights.
2.  **Data Preprocessing:** Cleaning the data, handling missing values, and encoding categorical features to prepare it for the model.
3.  **Feature Engineering:** Creating a robust preprocessing pipeline using `scikit-learn` to transform the data consistently.
4.  **Model Training:** Training a `RandomForestClassifier` on the prepared data.
5.  **Hyperparameter Tuning:** Using `GridSearchCV` to find the best parameters for the model to improve its accuracy.
6.  **Prediction & Submission:** Making predictions on the test set and generating a submission file in the format required by the Kaggle competition.

## workflow Project Workflow

The `prepared.ipynb` notebook follows these steps:

### 1\. Data Loading & Exploration

  - The training data (`train.csv`) is loaded into a pandas DataFrame.
  - A correlation heatmap is generated using `seaborn` to visualize relationships between numerical features like `Survived`, `Pclass`, `Age`, etc.

### 2\. Stratified Data Splitting

  - To ensure the training and test sets have a representative distribution of key features, `StratifiedShuffleSplit` is used. This is particularly important for maintaining the same proportion of survivors and passenger classes in both splits.

### 3\. Data Preprocessing Pipeline

A custom `scikit-learn` `Pipeline` is built to automate the feature engineering process. This pipeline consists of several custom transformers:

  - `AgeImputer`: Fills missing `Age` values using the mean.
  - `FeatureEncoder`: Converts categorical columns (`Embarked`, `Sex`) into numerical format using `OneHotEncoder`.
  - `FeatureDropper`: Removes unnecessary columns (`Name`, `Ticket`, `Cabin`, etc.) that are not useful for the model.

### 4\. Model Training & Hyperparameter Tuning

  - **Model Selection:** A `RandomForestClassifier` is chosen for this classification task.
  - **Scaling:** The feature data is scaled using `StandardScaler` to ensure all features contribute equally to the model's performance.
  - **Grid Search:** `GridSearchCV` is used to systematically test a range of hyperparameters (`n_estimators`, `max_depth`, `min_samples_split`) to find the optimal combination that yields the highest accuracy.

### 5\. Final Model & Prediction

  - The best estimator found by `GridSearchCV` is trained on the *entire* training dataset.
  - The same preprocessing pipeline is applied to the test data (`test.csv`).
  - The final trained model is used to predict survival outcomes for the test data.
    \-.  **Submission:** The predictions are saved to a `prediction.csv` file in the required format for submission to Kaggle.

## 🛠️ Technologies Used

  - **Python 3**
  - **Pandas:** For data manipulation and analysis.
  - **NumPy:** For numerical operations.
  - **Matplotlib & Seaborn:** For data visualization.
  - **Scikit-learn:** For building the machine learning pipeline, model training, and evaluation.
  - **Jupyter Notebook:** For interactive development and documentation.

## 🚀 How to Run

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/AjitChauhan081/Titanic-Survival-Prediction.git
    cd Titanic-Survival-Prediction
    ```

2.  **Create and activate a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Place the data files:**

      - Download `train.csv` and `test.csv` from the [Kaggle Titanic competition page](https://www.kaggle.com/c/titanic/data).
      - Create a `data` folder in the root directory and place the CSV files inside it.

5.  **Launch Jupyter Notebook:**

    ```bash
    jupyter notebook
    ```

6.  Open and run the `prepared.ipynb` notebook to see the complete workflow and generate the `prediction.csv` file.
