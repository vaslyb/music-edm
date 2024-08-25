import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import os 
import pandas as pd

## Train Model

# define model
model = LinearDiscriminantAnalysis(random_state=0)

# define model evaluation method
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

# fit model
model.fit(X, y)

# Predict class labels
y_pred = model.predict(X)

# Predict class probabilities
y_prob = model.predict_proba(X)

## Hyperparameter Tuning
# define grid
grid = dict()
grid['solver'] = ['svd', 'lsqr', 'eigen']
grid['shrinkage'] = np.arange(0, 1, 0.01)
# define search
search = GridSearchCV(model, grid, scoring='accuracy', cv=cv, n_jobs=-1)
# perform the search
# results = search.fit(X, y)
# # summarize
# print('Mean Accuracy: %.3f' % results.best_score_)
# print('Config: %s' % results.best_params_)

# Evaluate Model with Cross-Validation
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# summarize result
print('Mean Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

## Interpretations

os.makedirs('./results/lda', exist_ok=True)
feature_names = np.loadtxt('./dataset/data.csv', delimiter=',', skiprows=0, dtype=str, max_rows=1)
feature_names = [feature.replace('_', ' ').capitalize() for feature in list(feature_names)]
target_names = np.loadtxt('./dataset/labels.csv', delimiter=',', skiprows=0, dtype=str, max_rows=1)
target_names = [target.replace('_', ' ').capitalize() for target in list(target_names)]

# Explained variance (captures the ratio of the total variance each principal component captures, how whel each LD separates the classes)
print("Explained Variance Ratio:")
print(model.explained_variance_ratio_)

# Coefficients (In LDA, the coefficients are weights associated with each feature for constructing the Linear Discriminants. Each Linear Discriminant is a linear combination of the original features, and the coefficients determine the contribution of each feature to this combination.)
coefficients = model.coef_
coefficients_df = pd.DataFrame(coefficients, columns=feature_names, index=target_names)
coefficients_df.to_csv('./results/lda/coefficients.csv')

# Confusion Matrix
conf_matrix = confusion_matrix(y, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.matshow(conf_matrix, cmap='Blues')
# Add colorbar
plt.colorbar(cax)
# Add labels
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
# Add text annotations
for i in range(len(target_names)):
    for j in range(len(target_names)):
        ax.text(j, i, conf_matrix[i, j], ha='center', va='center', color='black')
# Set ticks and labels
ax.set_xticks(np.arange(len(target_names)))
ax.set_yticks(np.arange(len(target_names)))
ax.set_xticklabels(target_names)
ax.set_yticklabels(target_names)
# Rotate the tick labels and set their alignment
plt.xticks(rotation=90, ha='right')
# Adjust layout
plt.tight_layout()
# Save the plot
plt.savefig('./results/lda/confusion_matrix.png')  # Save as PNG file
plt.show()

# Classification report
class_report = classification_report(y, y_pred)
# print("Classification Report:")
# print(class_report)

# Plotting the LDs
X_r = model.transform(X)

# Get the number of LDs
n_lds = X_r.shape[1]

# Define colors and target names
colors = ['red', 'green', 'blue', 'purple']
target_names = [target.replace('_', ' ').capitalize() for target in list(target_names)]
# Plotting each pair of LDs
for i in range(n_lds):
    for j in range(i + 1, n_lds):
        plt.figure()
        for color, target_class in zip(colors, np.unique(y)):
            plt.scatter(X_r[y == target_class, i], 
                        X_r[y == target_class, j], 
                        alpha=0.8, color=color, label=target_names[target_class])
        plt.xlabel(f'LD {i + 1}')
        plt.ylabel(f'LD {j + 1}')
        plt.legend(loc='best')
        plt.title(f'LDA of Dataset: LD {i + 1} vs LD {j + 1}')
        plt.savefig(f'./results/lda/ld{i + 1}_vs_ld{j + 1}.png')