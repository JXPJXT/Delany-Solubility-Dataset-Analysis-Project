# Predicting Water Solubility of Organic Molecules using Delaney (ESOL) Dataset

## Overview

This project involves predicting the water solubility (log solubility in mols per liter) of common organic small molecules using the Delaney solubility dataset (ESOL) from MoleculeNet. The project employs linear regression and random forest regression models to achieve this prediction.

## Dataset

- **Name:** Delaney (ESOL) Dataset
- **Description:** Water solubility data for common organic small molecules.
- **Source:** MoleculeNet
- **Target Variable:** Log solubility (logS) in mols per liter

## Data Split

- **Training Set:** 80%
- **Test Set:** 20%

## Models Used

- **Linear Regression**
- **Random Forest Regression**

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/solubility-prediction.git
    cd solubility-prediction
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Data Preprocessing:**
    - Load the data:
        ```python
        import pandas as pd
        
        df = pd.read_csv("data/dataset.csv")
        print(df.head())
        ```
    - Select the target variable:
        ```python
        y = df['logS']
        ```

2. **Model Training:**
    - Train Linear Regression model:
        ```python
        from sklearn.linear_model import LinearRegression
        
        model_lr = LinearRegression()
        model_lr.fit(X_train, y_train)
        ```
    - Train Random Forest Regression model:
        ```python
        from sklearn.ensemble import RandomForestRegressor
        
        model_rf = RandomForestRegressor()
        model_rf.fit(X_train, y_train)
        ```

3. **Model Evaluation:**
    - Evaluate the performance of the models:
        ```python
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        y_pred_lr = model_lr.predict(X_test)
        y_pred_rf = model_rf.predict(X_test)
        
        print("Linear Regression MAE:", mean_absolute_error(y_test, y_pred_lr))
        print("Random Forest MAE:", mean_absolute_error(y_test, y_pred_rf))
        ```

## Results

The performance of the models is evaluated using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²) score. 

## Acknowledgments

- MoleculeNet for providing the dataset.
- The open-source community for their invaluable tools and libraries.

---

**Author:** [Japjot](https://www.linkedin.com/in/Japjot-SinghB)
