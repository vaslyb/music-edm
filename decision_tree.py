import numpy as np
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,GridSearchCV,RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import os 
import warnings
import graphviz

warnings.filterwarnings("ignore")

# Define model 
model = DecisionTreeClassifier(random_state=0)
if os.path.exists('./results/decision_tree/best_hyperparameters.txt'):
    best_params = {}
    with open('./results/decision_tree/best_hyperparameters.txt', 'r') as f:
        lines = f.readlines()
        best_score = float(lines[0].strip().split(': ')[1])
        for line in lines[2:]:
            param, value = line.strip().split(': ')
            if param == "ccp_alpha":
                value = float(value)
            if param == "max_depth" or param == "min_samples_split":
                value = int(value)
            best_params[param] = value
    model.set_params(**best_params)

# Define evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# Load the dataset
X = np.loadtxt('./dataset/data.csv', delimiter=',', skiprows=1)
y = np.loadtxt('./dataset/labels.csv', delimiter=',', skiprows=1)

# Replace infinite values with the maximum finite value or the minimum finite value
X = np.where(np.isposinf(X), np.nanmax(X[np.isfinite(X)]), X)
X = np.where(np.isneginf(X), np.nanmin(X[np.isfinite(X)]), X)

# Transform one hot encoded labels to integers
y = np.argmax(y, axis=1)

# Fit the model
model.fit(X, y)

# Model predictions
y_pred = model.predict(X)
y_prob = model.predict_proba(X)

# Hyperparameter Tuning with sklearn
if not os.path.exists('./results/decision_tree/best_hyperparameters.txt'):
    grid = dict()
    grid['criterion'] = ['gini', 'entropy']
    grid['max_depth'] = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    grid['min_samples_split'] = [2, 5, 10]
    grid['ccp_alpha'] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    search = GridSearchCV(model, grid, scoring='accuracy', cv=cv, n_jobs=-1)
    results = search.fit(X, y)
    best_params = results.best_params_
    best_score = results.best_score_
    with open('./results/decision_tree/best_hyperparameters.txt', 'w') as f:
        f.write('Mean Accuracy: %.3f\n' % best_score)
        f.write('Best Hyperparameters:\n')
        for param, value in best_params.items():
            f.write('%s: %s\n' % (param, value))
    model.set_params(**best_params)

# Evaluate Model with Cross-Validation with sklearn
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Mean Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

# Interpretations
os.makedirs('./results/decision_tree', exist_ok=True)

# Define the feature names and target names
feature_names = np.loadtxt('./dataset/data.csv', delimiter=',', skiprows=0, dtype=str, max_rows=1)
feature_names = [feature.replace('_', ' ').capitalize() for feature in list(feature_names)]
feature_names = [feature.replace(' mean', '') for feature in feature_names]
target_names = np.loadtxt('./dataset/labels.csv', delimiter=',', skiprows=0, dtype=str, max_rows=1)
target_names = [target.replace('_', ' ').capitalize() for target in list(target_names)]

# Visualize the Decision Tree
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
graph.render('./results/decision_tree/decision_tree', format='png', cleanup=True)

# Feature Importance
importances = model.feature_importances_
plt.figure(figsize=(10, 6))
sorted_indices = np.argsort(importances)
sorted_features = [feature_names[i] for i in sorted_indices]
sorted_importances = importances[sorted_indices]
plt.barh(sorted_features, sorted_importances, color='maroon')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances')
plt.tight_layout()
plt.savefig('./results/decision_tree/feature_importances.png')

top10 = np.argsort(importances)[::-1][:10]
top10_features = [feature_names[i] for i in top10][::-1]
top10_importances = importances[top10][::-1]
plt.figure(figsize=(10, 6))
plt.barh(top10_features, top10_importances, color='maroon')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 10 Feature Importances')
plt.tight_layout()
plt.savefig('./results/decision_tree/top10_feature_importances.png')

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
plt.savefig('./results/decision_tree/confusion_matrix.png')  
plt.show()

# Post pruning analysis

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
path = model.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# Plot the total impurity of the tree vs effective alpha for training set
fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")
fig.tight_layout()
fig.savefig('./results/decision_tree/total_impurity_vs_effective_alpha.png')

# Train decision trees using effective alphas
models = []
for ccp_alpha in ccp_alphas:
    model = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha, max_depth=int(best_params['max_depth']),min_samples_split=int(best_params['min_samples_split']))
    model.fit(X_train, y_train)
    models.append(model)
print(
    "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
        models[-1].tree_.node_count, ccp_alphas[-1]
    )
)
models = models[:-1]
ccp_alphas = ccp_alphas[:-1]

# Number of nodes and depth vs alpha
node_counts = [model.tree_.node_count for model in models]
depth = [model.tree_.max_depth for model in models]
fig, ax = plt.subplots(2, 1)
ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()
fig.savefig('./results/decision_tree/number_of_nodes_vs_alpha.png')

# Accuracy vs alpha for training and testing sets
train_scores = [model.score(X_train, y_train) for model in models]
test_scores = [model.score(X_test, y_test) for model in models]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker="o", drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker="o", drawstyle="steps-post")
ax.legend()
plt.savefig('./results/decision_tree/accuracy_vs_alpha.png')

# Keep the best model according to test set and save it

best_model_index = np.argmax(test_scores)
best_model = models[best_model_index]

dot_data = export_graphviz(
    best_model, 
    out_file=None, 
    feature_names=feature_names, 
    class_names=target_names, 
    filled=True,         # Set to False as per your request
    rounded=True,        # Rounded boxes for better aesthetics
    special_characters=True,
    proportion=True,     # Node sizes proportional to the number of samples
    precision=2          # Precision for node values
)
graph = graphviz.Source(dot_data)
graph.render('./results/decision_tree/best_decision_tree', format='png', cleanup=True)

# Classification report
class_report = classification_report(y, y_pred, target_names=target_names)
with open('./results/decision_tree/classification_report.txt', 'w') as f:
    f.write("Classification Report:\n")
    f.write(class_report)