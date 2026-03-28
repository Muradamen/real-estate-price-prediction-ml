
# 🏡 House Price Prediction Project

## Project Overview
This project focuses on building and evaluating linear regression models to predict house prices based on various property and neighborhood features. It covers the complete machine learning workflow, from exploratory data analysis (EDA) and data preprocessing to model building, assumption validation, regularization techniques, and model persistence.

## Problem Statement
A real-estate analytics team aims to estimate house prices in a city. This project addresses the challenge by:
1.  **Exploring the dataset** to understand feature relationships.
2.  Building and interpreting a **Multiple Linear Regression** model.
3.  **Validating core assumptions** of linear regression (linearity, independence, homoscedasticity, normality, multicollinearity, outliers).
4.  Applying **variable selection and regularization** techniques (Ridge and LASSO) to enhance model generalization and address multicollinearity.
5.  **Persisting the best-performing model** for future deployment.

## Key Features and Learnings
*   **Data Loading and Exploration (EDA):** Understanding dataset structure, summary statistics, and initial correlations.
*   **Multiple Linear Regression:** Building a baseline model using `sklearn.linear_model.LinearRegression`.
*   **Model Diagnostics:** Detailed analysis using `statsmodels.api.OLS` to evaluate coefficients, p-values, R-squared, and detect multicollinearity (VIF).
*   **Assumption Testing:** Visual and statistical checks for normality of residuals, homoscedasticity, and outliers.
*   **Regularization:** Implementing **Ridge (L2)** and **LASSO (L1)** regression to mitigate overfitting and manage multicollinearity, including scaling features.
*   **Model Evaluation:** Using metrics like MSE, RMSE, R-squared, and MAE for both training and test sets.
*   **Model Persistence:** Saving and loading the trained model and scaler using `joblib` for future use.

## Dataset
The project uses the `house_prices_portfolio.csv` dataset, which includes the following features:
-   **House_Price** (target, in $1,000s)
-   **Lot_Size** (m²)
-   **Bedrooms** (count)
-   **Bathrooms** (count)
-   **House_Age** (years)
-   **Distance_to_CityCenter** (km)
-   **Crime_Rate** (0–100 index)
-   **Nearby_Schools** (count)
-   **Monthly_Income** ($)
-   **Renovated** (0/1)
-   **Energy_Efficiency_Score** (0–100)
-   **Garden_Size** (m²)
-   **Noise_Level** (0–10)

## Technologies Used
*   Python 3.x
*   Pandas (for data manipulation)
*   NumPy (for numerical operations)
*   Scikit-learn (for linear models, train-test split, scaling, and metrics)
*   Statsmodels (for detailed OLS regression diagnostics)
*   Matplotlib (for plotting)
*   Seaborn (for enhanced visualizations)
*   Joblib (for model persistence)

## How to Run the Project
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Muradamen/real-estate-price-prediction-ml.git
    cd real-estate-price-prediction-ml
    ```
2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Jupyter Notebook:**
    Open the `house_price_prediction.ipynb` notebook in Jupyter or Google Colab and run all cells.
    ```bash
    jupyter notebook
    ```

## Results and Insights
*   The baseline Linear Regression model achieved an R-squared of approximately 0.69 on the training set and 0.53 on the test set, indicating room for improvement in generalization.
*   Multicollinearity was identified among several features (e.g., `Monthly_Income`, `Crime_Rate`, `Energy_Efficiency_Score`), which can affect the stability and interpretability of OLS coefficients.
*   Residual analysis revealed violations of normality and homoscedasticity assumptions, suggesting that standard errors and p-values from OLS should be interpreted with caution.
*   Ridge and LASSO regularization provided similar performance to the baseline model in terms of R-squared, with Ridge being marginally better on the test set. These techniques helped to stabilize coefficients and address multicollinearity.
*   Key positive drivers of house prices include `Bedrooms`, `Bathrooms`, `Monthly_Income`, and `Nearby_Schools`. Significant negative drivers are `Distance_to_CityCenter`, `House_Age`, and `Crime_Rate`.

## Future Work
*   **Feature Engineering:** Create new features (e.g., polynomial features, interaction terms) to capture non-linear relationships.
*   **Advanced Models:** Explore other regression models such as Gradient Boosting (XGBoost, LightGBM) or Random Forests for potentially better performance.
*   **Outlier Treatment:** Investigate and address the anomalous negative `House_Price` value and other potential outliers.
*   **Hyperparameter Tuning:** Systematically tune the `alpha` parameter for Ridge and LASSO using cross-validation.
*   **Heteroscedasticity Correction:** Implement techniques like weighted least squares or robust standard errors to address heteroscedasticity.

## Contributing
Feel free to fork this repository, open issues, or submit pull requests to improve the project.


👨‍💻 Author
Murad Amin 
LinkedIn: www.linkedin.com/in/muradamin
GitHub: https://github.com/Muradamen

## License
This project is licensed under the MIT License.

"
