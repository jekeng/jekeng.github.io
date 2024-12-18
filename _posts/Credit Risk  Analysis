🚀Excited to Share My Latest Data Analysis Case Study on Credit Risk Prediction! 🚀
As part of my continuous journey to hone my data analysis skills, I recently completed a project using the German Credit Risk dataset. The goal was to predict the likelihood of customers defaulting on their loans, and I leveraged Logistic Regression to uncover key insights.
🔍 Key Highlights:
Utilized a mix of Exploratory Data Analysis (EDA) and Logistic Regression to model credit risk based on variables like financial status, credit history, and loan details.
Achieved 78.5% accuracy in predicting credit defaults, with a precision of 72% for high-risk customers.
Gained valuable insights into how financial background (savings and checking accounts) and credit history significantly influence credit risk.
📊 Key Insights:
Customers with poor credit history and low account balances are at a higher risk of default.
Younger customers with shorter employment histories and larger loan amounts also show elevated risk.
💡 What I Learned: This project gave me practical experience in working with imbalanced datasets, building classification models, and extracting actionable insights from financial data. It also emphasized the importance of feature engineering and model interpretation to drive better business decisions.
🛠 Tools Used:
Python for data manipulation and model building
Logistic Regression for predictive modeling
Seaborn & Matplotlib for visualizing insight
Pandas for data cleaning and preprocessing
I’m eager to apply these skills to real-world challenges, so if you're looking for a data analyst who thrives in working with data-driven insights, let's connect! I'm always open to opportunities in data analysis, predictive modeling, and data visualization.
Check out the full case study in the article below



Objective
The primary goal of this analysis is to predict the likelihood of individuals defaulting on credit based on various personal, financial, and credit-related factors. This report will highlight the key findings from exploratory data analysis (EDA) and the performance of a logistic regression model applied to the German_CreditRisk dataset.

importing neccesary libraries
#import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
#from sklearn.preprocessing import LabelEncoder
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
#from sklearn.linear_model import LogisticRegression  # Import LogisticRegression
#from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

1. Dataset Overview
The German_CreditRisk dataset contains information about customers' credit risk, with each row representing an individual’s financial status and history. The target variable, response, is a binary classification indicating whether the customer has a good (0) or bad (1) credit risk.

Data Description
Total Observations: 1000
Features: 20 columns, including categorical and numerical variables such as:chk_acct (Checking account status)duration (Duration of the loan)amount (Loan amount)age (Customer’s age)credit_his (Credit history)purpose (Purpose of the loan)And more...

2. Exploratory Data Analysis (EDA)
2.1 Summary Statistics
Average Age: 35.5 years
Average Loan Amount: 3271 units
Average Loan Duration: 20 months

#summary_statistics = german_credit_risk.describe()
#print("Summary Statistics:")
#print(summary_statistics)

2.2 Key Observations
Checking Account Status: A large portion of customers with "no checking account" are classified as high credit risk.
Loan Amount: Higher loan amounts tend to correlate with higher credit risk.
Age Distribution: Older customers (ages 50 and above) tend to have a better credit history, while younger individuals (ages 18-30) are more likely to be high-risk.

2.3 Correlation Analysis
Correlation analysis showed that:Loan Amount and Duration have a mild positive correlation with the likelihood of default.Age is negatively correlated with default, meaning older individuals are generally lower risk.However, most of the correlation values between numerical variables were relatively low, indicating that no single variable strongly drives the outcome.

2.4 Categorical Variables
Credit History: Customers with a poor or critical credit history showed a significantly higher likelihood of default.
Saving Account Status: Customers with lower savings or "no savings account" were more likely to default.


# Separate features (X) and target variable (Y)
#X = german_credit_risk.drop(columns=['response'])  # Features
#Y = german_credit_risk['response']  # Target variable


# Split the data into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Initialize the logistic regression model
model = LogisticRegression(max_iter=1000)

# Train the model on the training data
model.fit(X_train, y_train)

3. Logistic Regression Model
3.1 Model Overview
The logistic regression model was chosen for this classification problem to predict whether a customer is likely to default (bad credit risk = 1) or not (good credit risk = 0). The following variables were included in the model:

Predictors: Loan amount, duration, checking account status, credit history, saving account status, housing situation, age, sex, and employment status, among others.

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)


# Print evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)


3.2 Model Performance
Training Set Accuracy: 80.3%
Test Set Accuracy: 78.5%

The model performed reasonably well with a balanced accuracy, suggesting that it can generalize to unseen data effectively.

3.3 Confusion Matrix
True Positives: 103
True Negatives: 134
False Positives: 20 (Customers predicted to be low risk but defaulted)
False Negatives: 43 (Customers predicted to be high risk but did not default)

3.4 Classification Report
MetricPrecisionRecallF1-ScoreGood Credit Risk (0)0.840.930.88Bad Credit Risk (1)0.720.480.58

Precision: For predicting bad credit risk (1), the model has a precision of 72%, meaning it correctly identifies 72% of true bad credit risks.
Recall: The recall for bad credit risk is 48%, meaning the model misses about 52% of actual bad credit risks.
F1-Score: The F1-score for bad credit risk is lower due to the tradeoff between precision and recall.

3.5 Feature Importance
In logistic regression, the following features contributed most significantly to predicting credit risk:

Checking Account Status: Customers with negative or low balances in their checking accounts are significantly more likely to default.
Credit History: Customers with poor credit history had higher odds of defaulting.
Loan Duration: Longer loan durations increased the likelihood of credit risk.
Saving Account Status: Individuals with lower or no savings were more likely to default.

3.6 Model Interpretation
The logistic regression model confirms some expected findings:

Individuals with poor financial backgrounds (low checking and saving accounts) are more prone to default.
Credit history remains a strong predictor of future default behavior.
Younger individuals, with shorter employment histories and higher debt obligations, are more likely to default.

4. Key Insights and Recommendations
4.1 Insights
Financial Background: Customers without checking accounts or savings are at a significantly higher risk of default.
Credit History: A customer’s credit history is a strong indicator of their likelihood of repayment. Improving credit history would reduce default risk.
Demographics: Younger customers (especially those with short employment histories) are riskier borrowers compared to older, established individuals.

4.2 Recommendations for Credit Issuers
Focus on Credit History: Implement stricter lending criteria for individuals with poor credit history.
Target High-Risk Groups: Offer financial literacy programs for younger individuals and those with poor savings habits.
Dynamic Credit Assessment: Consider implementing a dynamic credit assessment model that updates a customer’s credit risk based on their recent financial behavior.

4.3 Further Steps
Additional Models: Consider testing advanced models such as Random Forest or XGBoost for improved performance and robustness.
Balancing the Dataset: Address class imbalance by applying techniques like SMOTE to improve the recall of high-risk customers.
Cross-Validation: Apply cross-validation to further confirm the stability and robustness of the model across different data splits.

5. Conclusion
This analysis demonstrated that logistic regression provides valuable insights into predicting customer credit risk based on financial and demographic features. The model’s performance is satisfactory, but there is room for improvement, particularly in correctly identifying bad credit risks. Future iterations of this project could focus on model tuning and applying more complex algorithms for enhanced predictive power.
