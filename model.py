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
# But, the data is not standardized, so the coefficients are not directly comparable.

