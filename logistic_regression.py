import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score,GridSearchCV,RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import os 
import pandas as pd
from scipy.stats import chi2
import warnings
warnings.filterwarnings("ignore")

# Define model 
model = LogisticRegression(random_state=0, max_iter=1000, multi_class='multinomial')
if os.path.exists('./results/logistic_regression/best_hyperparameters.txt'):
    # Load the best hyperparameters from the file
    loaded_params = {}
    with open('./results/logistic_regression/best_hyperparameters.txt', 'r') as f:
        lines = f.readlines()
        best_score = float(lines[0].strip().split(': ')[1])
        for line in lines[2:]:
            param, value = line.strip().split(': ')
            loaded_params[param] = value
    # Use the loaded hyperparameters
    model.set_params(**loaded_params)

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

# transform one hot encoded labels to integers
y = np.argmax(y, axis=1)

# Fit the model
model.fit(X, y)

# model predictions
y_pred = model.predict(X)
y_prob = model.predict_proba(X)

# Hyperparameter Tuning with sklearn
if not os.path.exists('./results/logistic_regression/best_hyperparameters.txt'):
    grid = dict()
    grid['penalty'] = ['l1', 'l2', 'elasticnet', 'none']
    grid['solver'] = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    grid['C'] = [100, 10, 1.0, 0.1, 0.01]
    search = GridSearchCV(model, grid, scoring='accuracy', cv=cv, n_jobs=-1)
    results = search.fit(X, y)

    # Save the best hyperparameters
    best_params = results.best_params_
    best_score = results.best_score_

    # Save the best hyperparameters to a file
    with open('./results/logistic_regression/best_hyperparameters.txt', 'w') as f:
        f.write('Mean Accuracy: %.3f\n' % best_score)
        f.write('Best Hyperparameters:\n')
        for param, value in best_params.items():
            f.write('%s: %s\n' % (param, value))
    model.set_params(**best_params)

# Evaluate Model with Cross-Validation with sklearn
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Mean Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

# Interpretations
os.makedirs('./results/logistic_regression', exist_ok=True)

# Define the feature names and target names
feature_names = np.loadtxt('./dataset/data.csv', delimiter=',', skiprows=0, dtype=str, max_rows=1)
feature_names = [feature.replace('_', ' ').capitalize() for feature in list(feature_names)]
feature_names = [feature.replace(' mean', '') for feature in feature_names]
target_names = np.loadtxt('./dataset/labels.csv', delimiter=',', skiprows=0, dtype=str, max_rows=1)
target_names = [target.replace('_', ' ').capitalize() for target in list(target_names)]

# Coefficients
coefficients = model.coef_
coefficients_df = pd.DataFrame(coefficients, columns=feature_names, index=target_names)
coefficinets_df_transposed = coefficients_df.T
coefficients_df.to_csv('./results/logistic_regression/coefficients.csv')

# Design matrix 
X_design = np.hstack([np.ones((X.shape[0], 1)), X])

# Weight matrix
W = np.zeros((y_prob.shape[1],X_design.shape[0], X_design.shape[0]))
for i in range(X.shape[0]):  
    for j in range(y_prob.shape[1]):
        p = y_prob[i, j]
        W[j,i,i] = p * (1 - p) 
        
# Hessian matrix
Hessian = np.zeros((y_prob.shape[1],X_design.shape[1], X_design.shape[1]))
for j in range(y_prob.shape[1]):    
    Hessian[j] = np.dot(X_design.T, np.dot(W[j], X_design))
    
# Covariance matrix    
cov_matrix = np.linalg.inv(Hessian)

# Standard errors
standard_errors = np.zeros((y_prob.shape[1],X_design.shape[1]))
for j in range(y_prob.shape[1]):
    standard_errors[j] = np.sqrt(np.diag(cov_matrix[j]))
 
# Wald statistic 
logitParams = np.hstack([model.intercept_.reshape(-1, 1), model.coef_])
wald_statistics = (logitParams / standard_errors) ** 2
p_values = chi2.sf(wald_statistics, df=1)
with open('./results/logistic_regression/p_values_ward_test.csv', 'w') as f:
    f.write('Feature,Class,P-value\n')
    for i, target in enumerate(target_names):
        for j, feature in enumerate(feature_names):
            f.write(f'{feature},{target},{p_values[i, j]}\n')
            
# Plot all coefficients
plt.figure(figsize=(10, 6))
coefficinets_df_transposed.plot(kind='bar')
plt.ylabel('Coefficient Value')
plt.xlabel('Features')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('./results/logistic_regression/coefficients.png') 

# Plot the most important coefficients based on the feature importance
top_coefficients = np.argsort(np.mean(np.abs(coefficients), axis=0))[::-1]
top10 = top_coefficients[:10]
coeff_df_temp = coefficinets_df_transposed.iloc[top10]
plt.figure(figsize=(10, 6))
coeff_df_temp.plot(kind='bar')
plt.title('Coefficients per Class')
plt.ylabel('Coefficient Value')
plt.xlabel('Features')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('./results/logistic_regression/important_coefficients.png') 

# Plot the Effects
def plot_effects(effects, feature_names, target_name, output_dir, important=False):
    plt.figure()
    data = [effects[target_name][feature] for feature in feature_names]
    plt.boxplot(data, labels=feature_names, patch_artist=False, showmeans=True, meanline=True, showfliers=True)
    plt.title(f'{target_name.replace("_", " ").capitalize()}')
    plt.xticks(rotation=90) 
    plt.xlabel('Feature')
    plt.ylabel("Effect")
    plt.tight_layout()
    if important:
        plt.savefig(os.path.join(output_dir, f'{target_name}_important_effects_boxplot.png'))
    else:
        plt.savefig(os.path.join(output_dir, f'{target_name}_effects_boxplot.png'))
    plt.close()
n_classes, n_features = coefficients.shape
effects_dict = {}
for class_idx, class_name in enumerate(target_names):
    class_coefficients = coefficients[class_idx]
    effects_per_class = np.multiply(X , class_coefficients)  # Dot product
    for feature_idx in range(n_features):
        if class_name not in effects_dict:
            effects_dict[class_name] = {}
            if feature_names[feature_idx] not in effects_dict[class_name]:
                effects_dict[class_name][feature_names[feature_idx]] = []
        effects_dict[class_name][feature_names[feature_idx]] = effects_per_class[:, feature_idx]
for class_name in target_names:
    plot_effects(effects_dict, feature_names, class_name, './results/logistic_regression')

# Plot the most important effects based on the feature importance
top10_feauture_names = list(list(coeff_df_temp.to_dict().values())[0].keys())
for class_name in target_names:
    plot_effects(effects_dict, top10_feauture_names, class_name, './results/logistic_regression', important=True)

# Confusion Matrix
conf_matrix = confusion_matrix(y, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.matshow(conf_matrix, cmap='Blues')
plt.colorbar(cax)
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
for i in range(len(target_names)):
    for j in range(len(target_names)):
        ax.text(j, i, conf_matrix[i, j], ha='center', va='center', color='black')
ax.set_xticks(np.arange(len(target_names)))
ax.set_yticks(np.arange(len(target_names)))
ax.set_xticklabels(target_names)
ax.set_yticklabels(target_names)
plt.xticks(rotation=90, ha='right')
plt.tight_layout()
plt.savefig('./results/logistic_regression/confusion_matrix.png')  # Save as PNG file
plt.show()

# Classification report
class_report = classification_report(y, y_pred, target_names=target_names)
with open('./results/logistic_regression/classification_report.txt', 'w') as f:
    f.write("Classification Report:\n")
    f.write(class_report)