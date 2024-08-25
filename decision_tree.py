import numpy as np
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score,GridSearchCV,RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import os 
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Define model 
model = DecisionTreeClassifier(random_state=0)
if os.path.exists('./results/decision_trees/best_hyperparameters.txt'):
    loaded_params = {}
    with open('./results/decision_trees/best_hyperparameters.txt', 'r') as f:
        lines = f.readlines()
        best_score = float(lines[0].strip().split(': ')[1])
        for line in lines[2:]:
            param, value = line.strip().split(': ')
            if param == "ccp_alpha":
                value = float(value)
            if param == "max_depth" or param == "min_samples_split":
                value = int(value)
            loaded_params[param] = value
    model.set_params(**loaded_params)

# Define evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# Load the dataset
X = np.loadtxt('./dataset/data.csv', delimiter=',', skiprows=1)
y = np.loadtxt('./dataset/labels.csv', delimiter=',', skiprows=1)

# Replace infinite values with the maximum finite value or the minimum finite value
X = np.where(np.isposinf(X), np.nanmax(X[np.isfinite(X)]), X)
X = np.where(np.isneginf(X), np.nanmin(X[np.isfinite(X)]), X)

# Standardize the data
standarizer = StandardScaler()
X = standarizer.fit_transform(X)

# Transform one hot encoded labels to integers
y = np.argmax(y, axis=1)

# Fit the model
model.fit(X, y)

# Model predictions
y_pred = model.predict(X)
y_prob = model.predict_proba(X)

# Hyperparameter Tuning with sklearn
if not os.path.exists('./results/decision_trees/best_hyperparameters.txt'):
    grid = dict()
    grid['criterion'] = ['gini', 'entropy']
    grid['max_depth'] = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    grid['min_samples_split'] = [2, 5, 10]
    grid['ccp_alpha'] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    search = GridSearchCV(model, grid, scoring='accuracy', cv=cv, n_jobs=-1)
    results = search.fit(X, y)
    best_params = results.best_params_
    best_score = results.best_score_
    with open('./results/decision_trees/best_hyperparameters.txt', 'w') as f:
        f.write('Mean Accuracy: %.3f\n' % best_score)
        f.write('Best Hyperparameters:\n')
        for param, value in best_params.items():
            f.write('%s: %s\n' % (param, value))
    model.set_params(**best_params)

# Evaluate Model with Cross-Validation with sklearn
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Mean Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

# Interpretations
os.makedirs('./results/decision_trees', exist_ok=True)

# Define the feature names and target names
feature_names = np.loadtxt('./dataset/data.csv', delimiter=',', skiprows=0, dtype=str, max_rows=1)
feature_names = [feature.replace('_', ' ').capitalize() for feature in list(feature_names)]
feature_names = [feature.replace(' mean', '') for feature in feature_names]
target_names = np.loadtxt('./dataset/labels.csv', delimiter=',', skiprows=0, dtype=str, max_rows=1)
target_names = [target.replace('_', ' ').capitalize() for target in list(target_names)]

# Visualize the Decision Tree
import graphviz

# Export the decision tree to DOT format
dot_data = export_graphviz(
    model, 
    out_file=None, 
    feature_names=feature_names, 
    class_names=target_names, 
    filled=False,         # Set to False as per your request
    rounded=True,        # Rounded boxes for better aesthetics
    special_characters=True,
    proportion=True,     # Node sizes proportional to the number of samples
    precision=2          # Precision for node values
)
graph = graphviz.Source(dot_data)
graph.render('./results/decision_trees/decision_tree', format='png', cleanup=True)


# Confusion Matrix
conf_matrix = confusion_matrix(y, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.matshow(conf_matrix, cmap='YlGn')
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
plt.savefig('./results/decision_trees/confusion_matrix.png')  
plt.show()

# Classification report
class_report = classification_report(y, y_pred, target_names=target_names)
with open('./results/decision_trees/classification_report.txt', 'w') as f:
    f.write("Classification Report:\n")
    f.write(class_report)