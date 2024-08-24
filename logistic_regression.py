import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
import matplotlib.pyplot as plt
import os 
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

# Define model 
model = LogisticRegression(random_state=0, max_iter=1000,multi_class='multinomial')

# Define evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# Load the dataset
X = np.loadtxt('./dataset/data.csv', delimiter=',', skiprows=1)
y = np.loadtxt('./dataset/labels.csv', delimiter=',', skiprows=1)

# replace infinite values with the maximum finite value or the minimum finite value
X = np.where(np.isposinf(X), np.nanmax(X[np.isfinite(X)]), X)
X = np.where(np.isneginf(X), np.nanmin(X[np.isfinite(X)]), X)

# standardize the data
standarizer = StandardScaler()
X = standarizer.fit_transform(X)

# Define model
model = sm.MNLogit(y, X)

# Fit model
result = model.fit(maxiter=1000)

# transform one hot encoded labels to integers
y = np.argmax(y, axis=1)

# model predictions
y_pred = np.argmax(model.predict(result.params, X), axis=1)
y_prob = model.predict(result.params, X)

#  Hyperparameter Tuning with sklearn
# grid = dict()
# grid['penalty'] = ['l1', 'l2', 'elasticnet', 'none']
# grid['solver'] = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
# grid['C'] = [100, 10, 1.0, 0.1, 0.01]
# search = GridSearchCV(model, grid, scoring='accuracy', cv=cv, n_jobs=-1)
# results = search.fit(X, y)
# print('Mean Accuracy: %.3f' % results.best_score_)
# print('Config: %s' % results.best_params_)

# Evaluate Model with Cross-Validation with sklearn
# scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# print('Mean Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

# Interpretations

os.makedirs('./results/logistic_regression', exist_ok=True)

# Define the feature names and target names
feature_names = np.loadtxt('./dataset/data.csv', delimiter=',', skiprows=0, dtype=str, max_rows=1)
feature_names = [feature.replace('_', ' ').capitalize() for feature in list(feature_names)]
feature_names = [feature.replace(' mean', '') for feature in feature_names]
feature_names = ['Intercept'] + feature_names
target_names = np.loadtxt('./dataset/labels.csv', delimiter=',', skiprows=0, dtype=str, max_rows=1)
target_names = [target.replace('_', ' ').capitalize() for target in list(target_names)]

# Coefficients
coefficients = result.params
inferred_coefficients = -coefficients.sum(axis=1)
coefficients = np.concatenate((coefficients, inferred_coefficients[:, None]), axis=1)
coefficients = coefficients.transpose()
coefficients_df = pd.DataFrame(coefficients, columns=feature_names, index=target_names)
coefficients_df.to_csv('./results/logistic_regression/coefficients.csv')

# Standard errors
standard_errors = result.bse
standard_errors = standard_errors.transpose()
# Calculate Wald statistics
wald_statistics = (coefficients / standard_errors) ** 2

# Calculate p-values
p_values = 1 - sm.stats.chisqprob(wald_statistics, df=1)

# Print the summary (includes coefficients, standard errors, and more)
print(result.summary())

# Get the confidence intervals (default is 95% CI)
conf_intervals = result.conf_int()
print("Confidence Intervals:\n", conf_intervals)

# Plot all coefficients
plt.figure(figsize=(10, 6))
coefficients_df.plot(kind='bar')
plt.title('Logistic Regression Coefficients for Each Class')
plt.ylabel('Coefficient Value')
plt.xlabel('Features')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('./results/logistic_regression/coefficients.png') 

# Plot the most important coefficients based on the feature importance

top_coefficients = np.argsort(np.mean(np.abs(coefficients), axis=0))[::-1]
coeff_df_temp = coefficients_df.iloc[top_coefficients[:10]]

plt.figure(figsize=(10, 6))
coeff_df_temp.plot(kind='bar')
plt.title('Logistic Regression Coefficients for Each Class for the Most Important Features')
plt.ylabel('Coefficient Value')
plt.xlabel('Features')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('./results/logistic_regression/important_coefficients.png') 


# Effect 
# print(X.shape, coefficients.T.shape)
# effects = np.dot(X, coefficients.T)  # X_test (samples x features) * coefficients (features x classes)

# # Create a DataFrame to store the effects
# effects_df = pd.DataFrame(effects, columns=target_names)

# # Save the effects to a CSV file
# output_path = './results/logistic_regression/effects.csv'
# effects_df.to_csv(output_path, index=False)

# # Confusion Matrix
# conf_matrix = confusion_matrix(y, y_pred)
# fig, ax = plt.subplots(figsize=(8, 6))
# cax = ax.matshow(conf_matrix, cmap='Reds')
# # Add colorbar
# plt.colorbar(cax)
# # Add labels
# ax.set_xlabel('Predicted Label')
# ax.set_ylabel('True Label')
# ax.set_title('Confusion Matrix')
# # Add text annotations
# for i in range(len(target_names)):
#     for j in range(len(target_names)):
#         ax.text(j, i, conf_matrix[i, j], ha='center', va='center', color='black')
# # Set ticks and labels
# ax.set_xticks(np.arange(len(target_names)))
# ax.set_yticks(np.arange(len(target_names)))
# ax.set_xticklabels(target_names)
# ax.set_yticklabels(target_names)
# # Rotate the tick labels and set their alignment
# plt.xticks(rotation=90, ha='right')
# # Adjust layout
# plt.tight_layout()
# # Save the plot
# plt.savefig('./results/logistic_regression/confusion_matrix.png')  # Save as PNG file
# plt.show()

# Classification report
class_report = classification_report(y, y_pred, target_names=target_names)
print("Classification Report:")
print(class_report)