#%%
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from EDAcopy import df_withdummies

# check the df_withdummies
print(df_withdummies.head())
print(df_withdummies.tail())
print(df_withdummies.shape)

#%%
# Select the relevant variables for modeling
X = df_withdummies[['year', 'c_temp', 'rainfall', 'snowfall', 'el_nino', 'g_temp', 'storms', 'disasters', 'spending']]  # Independent variables
y = df_withdummies['happening']  # Dependent variable

#%%
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add a constant term to the independent variables
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

#%%
# Fit the probit model
probit_model = sm.Probit(y_train, X_train_sm)
probit_results = probit_model.fit_regularized(method='l1', alpha=0.1)

# Make predictions on the testing set
y_pred_prob = probit_results.predict(X_test_sm)
y_pred = (y_pred_prob > 0.5).astype(int)

# Calculate accuracy, confusion matrix, and classification report
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Print the model summary
print(probit_results.summary())

#%%
import matplotlib.pyplot as plt

# Get the variable names and coefficients
var_names = X_train_sm.columns[1:]  # Exclude the constant term
coef = probit_results.params[1:]

#%%
# Create a bar plot of the coefficients
plt.figure(figsize=(10, 6))
plt.bar(var_names, coef)
plt.xticks(rotation=90)
plt.xlabel('Variables')
plt.ylabel('Coefficients')
plt.title('Probit Model Coefficients')
plt.tight_layout()
plt.show()

#%% [markdown]
# ### Interpretation of the Probit Model
# The Probit model stopped at 1 iteration with the inequality constraints incompatible message.
# The accuracy of the model is 0.318, which is very low. 
# The confusion matrix shows that the model is predicting all observations as not happening. 
# This is likely due to the imbalanced nature of the dataset as we observed in the EDA
# We observed that the number of observations where the perception that climate change is happening is much higher than the number of observations where the event is not happening.
# To address this, we will try using Firth's logistic regression model, which is known to perform better with imbalanced datasets.

# %%
# Fit Firth's logistic regression model
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
import seaborn as sns

# Select the relevant variables for modeling
X = df_withdummies[['year', 'c_temp', 'rainfall', 'snowfall', 'el_nino', 'g_temp', 'storms', 'disasters', 'spending']]  # Independent variables
y = df_withdummies['happening']  # Dependent variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Firth's logistic regression model
firth_model = LogisticRegressionCV(penalty='l1', solver='saga', cv=5, random_state=42, max_iter=1000)
firth_model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = firth_model.predict(X_test)

# Calculate accuracy, confusion matrix, and classification report
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Print the model coefficients
print("Model Coefficients:")
for feature, coef in zip(X.columns, firth_model.coef_[0]):
    print(f"{feature}: {coef}")

#%%
import pandas as pd
# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', cbar=False,
            xticklabels=['Predicted Not Happening', 'Predicted Happening'],
            yticklabels=['Actual Not Happening', 'Actual Happening'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Plot the coefficients
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': firth_model.coef_[0]})
coef_df = coef_df.sort_values('Coefficient', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=coef_df)
plt.xlabel('Coefficient')
plt.ylabel('Feature')
plt.title('Model Coefficients')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Interpretation of Firth's Logistic Regression Model
# The model's accuracy is 0.682, which is an improvement over the Probit model.
# However, the confusion matrix shows that the model is no better than the Probit model.
# As this model only predicts all observations as happening. Since, the dataset is imbalanced, the accuracy score looks improved.
# The model coefficients show that the 'spending' variable has the highest positive coefficient, indicating that it has the most significant impact on the prediction.
# But, even if the data is standardized, so the coefficients are not directly comparable.

# %%
# Standardize the data
from sklearn.preprocessing import StandardScaler

# Standardize the independent variables
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit Firth's logistic regression model on the standardized data
firth_model_scaled = LogisticRegressionCV(penalty='l1', solver='saga', cv=5, random_state=42, max_iter=1000)
firth_model_scaled.fit(X_train_scaled, y_train)

# Make predictions on the testing set
y_pred = firth_model_scaled.predict(X_test_scaled)

# Calculate accuracy, confusion matrix, and classification report
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Print the model coefficients
print("Model Coefficients:")
for feature, coef in zip(X.columns, firth_model_scaled.coef_[0]):
    print(f"{feature}: {coef}")
    
# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', cbar=False,
            xticklabels=['Predicted Not Happening', 'Predicted Happening'],
            yticklabels=['Actual Not Happening', 'Actual Happening'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Plot the coefficients
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': firth_model_scaled.coef_[0]})
coef_df = coef_df.sort_values('Coefficient', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=coef_df)
plt.xlabel('Coefficient')
plt.ylabel('Feature')
plt.title('Model Coefficients')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Interpretation of Firth's Logistic Regression Model on scaled data.
# The model's accuracy is still 0.682, which is not an improvement over the original Logit model.
# Even, the confusion matrix shows that the model is just the same as the original Logit model.
# As this model only predicts all observations as happening. Since, the dataset is imbalanced, the accuracy score looks improved.
# The model coefficients show that the 'spending' variable has the highest positive coefficient, indicating that it has the most significant impact on the prediction.
# But, even if the data is standardized, so the coefficients are not directly comparable.

#%%
# Fit the logistic regression model with PCA components
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from EDAcopy import dummy_PCA2

# Assuming you have the 'happening' variable in the original dataframe
y = df_withdummies['happening']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dummy_PCA2, y, test_size=0.2, random_state=42)

# Create and fit the logistic regression model
logit_model = LogisticRegression(random_state=42)
logit_model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = logit_model.predict(X_test)

# Calculate accuracy, confusion matrix, and classification report
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', cbar=False,
            xticklabels=['Predicted Not Happening', 'Predicted Happening'],
            yticklabels=['Actual Not Happening', 'Actual Happening'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Visualize the ROC curve
from sklearn.metrics import roc_curve, auc

y_pred_prob = logit_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Interpretation of Logistic Regression Model with PCA components
# The model's accuracy is 1.0 which is a perfect score.
# The confusion matrix shows that the model is predicting all observations correctly.
# The ROC curve shows that the model has a perfect AUC score of 1.0.
# This indicates that the model is performing exceptionally well on the testing set.
# - Now let's see check if the model is overfitting by checking the model performance on the training set.

#%%
# Make predictions on the training set
y_pred_train = logit_model.predict(X_train)

# Calculate accuracy, confusion matrix, and classification report on the training set
accuracy_train = accuracy_score(y_train, y_pred_train)
conf_matrix_train = confusion_matrix(y_train, y_pred_train)
class_report_train = classification_report(y_train, y_pred_train)

print("Accuracy (Training Set):", accuracy_train)
print("Confusion Matrix (Training Set):\n", conf_matrix_train)
print("Classification Report (Training Set):\n", class_report_train)

# Visualize the confusion matrix for the training set
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_train, annot=True, cmap='Blues', fmt='d', cbar=False,
            xticklabels=['Predicted Not Happening', 'Predicted Happening'],
            yticklabels=['Actual Not Happening', 'Actual Happening'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (Training Set)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Interpretation of Logistic Regression Model with PCA components on the training set
# The model's accuracy is 1.0 which is a perfect score.
# The confusion matrix shows that the model is predicting all observations correctly.
# This indicates that the model is performing exceptionally well on the training set as well.
# The model is not overfitting as it is performing well on both the training and testing sets.
# We can further check the model performance using cross-validation.

#%%
# Perform cross-validation to evaluate the model
from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation
cv_scores = cross_val_score(logit_model, dummy_PCA2, y, cv=5)

print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

# %% [markdown]
# ### Interpretation of Logit Model with PCA components using Cross-Validation
# The cross-validation scores are all 1.0, indicating that the model is performing exceptionally well on all folds.
# The mean cross-validation accuracy is also 1.0, which is a perfect score.
# This further confirms that the model is performing well and is not overfitting.