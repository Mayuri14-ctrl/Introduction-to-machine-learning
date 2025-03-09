# Introduction to Machine Learning

## Linear Regression

### Assumptions of Linear Regression

1. **Linear Relationship between Target and Independent Variables**
   - The relationship between the dependent and independent variables should be linear.
   - How to check:
     - Scatter plots or residual plots.
   - How to fix:
     - Apply log transformation to reduce variation or handle exponential relationships.

2. **No Multicollinearity Between Predictor Variables**
   - Why is it important?
     1. Multicollinearity makes the model unstable and sensitive to small variations in data.
     2. The model cannot distinguish the individual effect of each predictor.
     3. Inflated standard errors can lead to unreliable statistical inferences.
     4. If multicollinearity is too high, it will fail to invert the matrix, and the equation cannot be solved.
   - How to detect:
     - **Variance Inflation Factor (VIF):** Measures how much variance of a coefficient is inflated due to collinearity.
       - VIF > 5 → High multicollinearity (Consider removing the variable).
       - VIF > 10 → Severe multicollinearity (Must take action).
     - **Correlation Matrix:** A heatmap of correlations can reveal highly correlated variables.
   - How to solve:
     - Principal Component Analysis (PCA)
     - Regularization techniques (Lasso, Ridge)
     - Increase sample size
     - Remove one of the correlated variables

3. **Homoscedasticity**
   - The variance of residuals (errors) should be constant across all values of X.
   - Why is it important?
     - If violated, it could lead to biased standard errors → wrong conclusions in hypothesis testing.
   - How to check:
     - Residual plots (should show no clear pattern).

4. **Normality of Residuals**
   - The residuals should be normally distributed.
   - Why is it important?
     - Needed for reliable hypothesis testing and confidence intervals.
   - How to check:
     - Histogram or Q-Q Plot of residuals.
     - Shapiro-Wilk test.
   - How to fix:
     - Apply transformations like log, square root, or Box-Cox transformation.

5. **Independence of Residuals**
   - Residuals should not be correlated with each other.
   - Why is it important?
     - If violated, it can lead to biased coefficients and incorrect standard errors, impacting statistical tests and leading to false confidence in results.
   - How to check:
     - Durbin-Watson test (value close to 2 indicates no autocorrelation).

6. **No Endogeneity**
   - Independent variables should not be correlated with the error term.
   - Why is it important?
     - If violated, the model parameters will be biased and inconsistent.
   - How to check:
     - Use instrumental variables or perform the Hausman test.

### Loss Function
- **Mean Squared Error (MSE):** Measures the average squared difference between actual and predicted values.
- **Mean Absolute Error (MAE):** Measures the average absolute difference between actual and predicted values.

### Evaluation Metrics
- **R-squared (R²):** Explains the proportion of variance in the dependent variable explained by the model.
- **Adjusted R-squared:** Adjusts R² for the number of predictors.
- **Root Mean Squared Error (RMSE):** Measures model accuracy by penalizing large errors more than MAE.

---

## Logistic Regression

### Overview
- A supervised learning algorithm for binary classification.
- It predicts probability using the sigmoid function:
  
  \[ P(y|X) = \frac{1}{1 + e^{-X}} \]
  
### Why is it Called Regression if It's Used for Classification?
- It models the probability of an event using regression techniques.
- It predicts log-odds (a continuous value) and converts it to probability using the sigmoid function.

### What does the log-odds (logit) function mean in logistic regression?
- It transforms probabilities into log odds to maintain linear relationship between independent variables and the log-odds.

### How do we interpret the coefficients in logistic regression?
- It represent Change in log odds of target variable for a unit change in predictor variable
- Exponentiating the coefficient gives the odds ratio.
  
### Why can't we use Mean Squared Error (MSE) as a loss function in logistic regression
- MSE leads to non convex loss function, it may not converge
- We use log - loss( binary cross entropy)

### What is log loss (binary cross-entropy), and how is it used for optimization?
- loss loss= -1/N sum(y log y + (1-y) log (1-y) )
- Penalizes incorrect predictions more heavily.

### What optimization algorithm is used to train logistic regression?
Gradient Descent (most common)
Newton’s Method (used in some cases for faster convergence)

### What are the key assumptions of logistic regression?
- No multicollinearity between independent variables.
- Linear relationship between independent variables and log-odds.
- Independent observations (no autocorrelation).
### What is multicollinearity? How does it affect logistic regression?
- Multicollinearity occurs when predictor variables are highly correlated.
- It makes the model unstable because coefficients become unreliable.
- Solution: Use Variance Inflation Factor (VIF) to detect multicollinearity and remove highly correlated variables.

### What is the problem of imbalanced data in logistic regression? How do you handle it?
- Undersampling/oversampling (SMOTE)
- Use class weights in loss function
- Decision tree
### How do outliers affect logistic regression?
-Outliers can skew the decision boundary because logistic regression is sensitive to extreme values.
- log transformation
- Winsorize the data
-  Apply regularization (L1/L2) to reduce their impact.
### Why do we need regularization (L1/L2) in logistic regression?
- To prevent overfitting by penalizing large coefficients.
- L1 (Lasso) → Feature selection by setting some coefficients to zero.
- L2 (Ridge) → Shrinks coefficients but does not make them zero.

## Performance Evaluation & Metrics
1️⃣8️⃣ What metrics do we use to evaluate a logistic regression model?
✅ Accuracy → 
Correct Predictions
Total Predictions
Total Predictions
Correct Predictions

✅ Precision → 
TP
TP + FP
TP + FP
TP
​
✅ Recall → 
TP
TP + FN
TP + FN
TP
​
✅ F1-score → Harmonic mean of precision and recall
✅ AUC-ROC → Measures how well the model separates classes.

### What is the ROC curve, and how do we interpret it?
The ROC curve plots True Positive Rate (TPR) vs. False Positive Rate (FPR).
A higher AUC (closer to 1) means a better model.
### How do we choose the right threshold for classification?
Default: 0.5, but it depends on the problem.
Use Precision-Recall tradeoff to find the best threshold.

### How does logistic regression handle categorical variables?
One-Hot Encoding for nominal variables.
Ordinal Encoding for ordered categories.
### You trained a logistic regression model, but it has low accuracy. What would you do?
✅ Check for missing values.
✅ Try feature engineering.
✅ Tune regularization strength.
✅ Handle class imbalance.


---


### How to Contribute
- Fork this repository.
- Create a new branch (`git checkout -b feature-branch`).
- Commit your changes (`git commit -m "Added explanation on assumptions"`).
- Push to the branch (`git push origin feature-branch`).
- Open a pull request!

---

### License
This project is licensed under the MIT License.



### SVM
### K-nearest neighbours
### K-means
### Decision tree
### Bagging
### Boosting
