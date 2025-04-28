# linear-regression
# Linear Regression Task



## Objective

The objective of this task is to implement and understand simple and multiple linear regression using Python libraries.

## Tools Used

* Scikit-learn
* Pandas
* Matplotlib

## Dataset

The code is designed to work with a dataset relevant to regression. A common example, and the one mentioned in the task, is a House Price Prediction Dataset.

**Note:** You need to replace `'house_price_dataset.csv'`, `'target_column'`, and `'feature_columns'` in the code with the actual filename and column names from the dataset you are using.

## Code Description

The Python script `linear_regression.py` (or whatever you name your file) performs the following steps:

1.  **Import Libraries:** Imports necessary libraries for data handling, model training, evaluation, and plotting.
2.  **Load and Preprocess Data:** Reads the dataset from a CSV file into a pandas DataFrame. Includes basic preprocessing for handling missing values (by replacing them with the mean). You may need to adapt preprocessing based on your specific dataset.
3.  **Split Data:** Splits the dataset into training and testing sets to evaluate the model's performance on unseen data.
4.  **Fit Linear Regression Model:** Trains a Linear Regression model using the scikit-learn library on the training data.
5.  **Evaluate Model:** Evaluates the trained model on the test set using common regression metrics:
    * Mean Absolute Error (MAE)
    * Mean Squared Error (MSE)
    * R-squared (R²)
6.  **Plot and Interpret:**
    * For simple linear regression (a single feature), it generates a scatter plot of the test data and overlays the fitted regression line.
    * Prints the intercept and coefficients of the fitted model. These values help in understanding the relationship between the features and the target variable.

## How to Run the Code

1.  Save the provided Python code as a `.py` file (e.g., `linear_regression.py`).
2.  Place your dataset file (e.g., `house_price_dataset.csv`) in the same directory as the Python script, or update the code with the correct path to your dataset.
3.  **Modify the code:**
    * Update the `pd.read_csv()` call with the correct filename of your dataset.
    * Replace `'target_column'` with the name of the column you want to predict.
    * Replace `'feature_columns'` with a list of the names of the feature columns you will use for prediction.
4.  Ensure you have the required libraries installed (`pandas`, `scikit-learn`, `matplotlib`). You can install them using pip:
    ```bash
    pip install pandas scikit-learn matplotlib
    ```
5.  Run the Python script from your terminal:
    ```bash
    python linear_regression.py
    ```

The script will output the evaluation metrics (MAE, MSE, R²) and the model's intercept and coefficients. If you are performing simple linear regression, a plot of the regression line will also be displayed.

## Interpretation

* **Coefficients:** The coefficients indicate the change in the target variable for a one-unit increase in the corresponding feature, holding other features constant.
* **Intercept:** The intercept is the predicted value of the target variable when all features are zero.
* **MAE, MSE, R²:** These metrics provide insights into how well the model fits the data. Lower MAE and MSE indicate better accuracy, while R² represents the proportion of the variance in the target variable that is predictable from the features.

