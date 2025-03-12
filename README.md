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
     - If violated, it could lead to biased standard errors → wrong conclusions in hypothesis testing and confidence interval.
   - How to check:
     - Residual plots (should show no clear pattern).
   - Solution: Use log transformation or Weighted Least Squares (WLS

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

### Why does Linear Regression minimize the sum of squared errors (SSE) instead of absolute errors?
1. Amplifies effect of larger error
2. Squared errors are differentiable, making optimization easier
3. Single global minimum, absolute errors create multiple local minima

### Why is Linear Regression sensitive to outliers? How do you handle them?
- OLS squares the error, leading to high penalization of large errors
- How to Detect Outliers?
  - Boxplots & Scatter plots: Identify extreme values.
  - Z-score Method: If ∣Z∣>3, it’s an outlier.
- Ways to Handle Outliers
  - Remove extreme outliers if they are data entry errors.
  - Transform variables (logarithm or square root).

### What is 𝑅2, and why does adding more variables always increase its value?
- measures how well independent variables explain the variance in
- Adding variables always increases or keeps it constant because residuals shrink.

### What happens if the matrix 𝑋𝑇 X is not invertible?
- Multicollinearity: Strong correlation among predictors.
- Too many features relative to observations.

### What are the optimization Algorithm for Linear Regression?
- OLS : It minimises sum of squared error,
   - it find reression coefficient using normal equation
     \[
\hat{\beta} = (X^T X)^{-1} X^T Y
\]
   - Used when small dataset, Fast and gives exact solutions.
- Gradient Descent:
   - It iteratively updates the regression coefficients by minimizing the cost function.
   - It uses the partial derivative of the cost function to adjust the weights step by step
   - Cost Function (Mean Squared Error - MSE)
   - \[J(\beta) = \frac{1}{2m} \sum_{i=1}^{m} (Y_i - \hat{Y}_i)^2\]

Where:
- **J(β)** = Cost function (Mean Squared Error)
- **m** = Number of samples
- **Yᵢ** = Actual values
- **Ŷᵢ** = Predicted values

#### Gradient Descent formula
\beta_j = \beta_j - \alpha \frac{\partial J}{\partial \beta_j}
\]

Where:
- **βⱼ** = Parameter to be updated
- **α** = Learning rate (step size)
- **J** = Cost function

This equation helps in iteratively updating **βⱼ** to minimize the cost function **J**.

### What are the different types of Gradient Descent?
- Batch GD- It takes entire data and updates the parameters.Converges smoothly but is slow for large datasets.
- Stochastic GD- Updates coefficients using one data point at a time.Faster but has high variance
- Mini Batch GD- Updates coefficients using small batch of data point at a time.Balances speed and stability.
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
### What metrics do we use to evaluate a logistic regression model? 
- Accuracy → Correct Predictions / Total Predictions
- Precision →  TP /TP + FP
​- Recall →  TP/ TP + FN
​- F1-score → Harmonic mean of precision and recall
- AUC-ROC → Measures how well the model separates classes.

### What is the ROC curve, and how do we interpret it?
The ROC curve plots True Positive Rate (TPR) (Recall) vs. False Positive Rate (FPR).
A higher AUC (closer to 1) means a better model.

### How do we choose the right threshold for classification?
Default: 0.5, but it depends on the problem.
Use Precision-Recall tradeoff to find the best threshold.

### How does logistic regression handle categorical variables?
One-Hot Encoding for nominal variables.
Ordinal Encoding for ordered categories.

### You trained a logistic regression model, but it has low accuracy. What would you do?
- Check for missing values.
- Try feature engineering.
- Tune regularization strength.
- Handle class imbalance.

---
## Feature Engineering
### what is ordinal vs nominal data?
Ordinal Data (Ordered Categories)
The categories follow a logical order.
Best Encoding Approach:
Label Encoding (Ordinal Encoding) is used since order matters.
Nominal Data (Unordered Categories)
The categories do not have a meaningful order.
Best Encoding Approach:
One-Hot Encoding is preferred, creating binary columns:

### Min-Max Scaling vs. Standardization (Z-score Normalization)


---
## SVM
It can handle high dimensional data and create decision boundaries that separate data points of different classes
- Hyperplane : SVM finds the best hyperplane that separates the data points of different classes
- Maximizes the margin: It maximizes the margin between the closest data points from both classes
- Kernel: If the data is not linearly separable: SVM uses kernal function to transform the data into higher dimensional space where it becomes separable
Popular kernals include
Linear kernel
Polynomial kernel
RBF kernel
Sigmoid kernel

---
## K-nearest neighbours 
It is non parametric and instance based ,meaning no assumption of data distribution and stores all the training data instead of building explicit model
- no traning phase
- choosing right k is crucial (k too low - overfitting, k too high underfitting
- Decide no of k, for any new data point, find nearest k neighbours based on the smallest distance(Euclidean distance),Assign the mejority class label
- Computationally heavy
- Curse of Dimensionality
- Imbalanced data

### How does the choice of k affect the bias-variance tradeoff in KNN? 
A small (k=1) results in high variance and low bias (overfitting). The model fits noise and memorizes training data.
A large (k=n) results in high bias and low variance (underfitting). The model becomes too simplistic, ignoring complex patterns.
Best practice: Choose k using cross-validation or elbow method (finding the optimal tradeoff).

### How does feature scaling affect KNN?
Answer:
Problem: KNN relies on distance calculations (e.g., Euclidean distance). If features have different scales (e.g., age in years vs income in lakhs), higher-magnitude features dominate distance calculations.
Solution:
Min-Max Scaling: Normalizes data to [0,1].
Standardization (Z-score normalization): Converts data to have mean 0 and variance 1
Use Manhattan or Cosine Distance instead of Euclidean if feature magnitudes vary widely.

### What is the time complexity of KNN? How can it be improved?
Brute Force KNN (Naïve implementation):
Training time: O(1) (since no training is needed).
Prediction time: O(n⋅d), where n is the number of points and 
d is the number of dimensions.

Optimizations:
KD-Tree: Reduces search time to O(logn), but works best for 𝑑<30
Ball Tree: Performs better for high-dimensional data.
Approximate Nearest Neighbors (ANN): Uses hashing or clustering to reduce search space.

### What are the different distance metrics in KNN? When should each be used
Euclidean Distance:  Best for continuous numerical data with similar scales.
Manhattan Distance: Works well when movement is restricted to grid-like paths (e.g., city block distances).
Minkowski Distance: Generalized form of Euclidean and Manhattan:
Cosine Similarity: Measures the cosine of the angle between two vectors.Useful for text analysis (TF-IDF vectors) or high-dimensional sparse data.
Hamming Distance: Measures dissimilarity in categorical data.Used for binary or categorical features.

### How do you handle categorical variables in KNN?
Convert categories into numerical values using techniques like:
One-hot encoding
Label encoding
Use appropriate distance metrics:
Hamming Distance for binary variables
Gower Distance for mixed categorical and numerical data
Weighted Voting: Assign higher weights to neighbors that are more similar in categorical features

### Can KNN be used for probability estimation? If yes, how?
Relative Frequency Approach:
P(y = c | x) = \frac{\text{count of class } c \text{ in k-neighbors}}{k} ]
Weighted KNN: Assign higher weights to closer neighbors using inverse distance weighting:
w_i = \frac{1}{d(x, x_i) + \epsilon} ]
Probability is then calculated as a weighted sum of class occurrences.

---
## K-means
---
## Decision tree
---
## Bagging
---
## Boosting
---
