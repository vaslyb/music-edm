import xgboost as xgb
import pandas as pd
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import json
import matplotlib.pyplot as plt
import shap
from sklearn.inspection import permutation_importance
import os

# Load the dataset
X = np.loadtxt('../../dataset/data.csv', delimiter=',', skiprows=1)
y = np.loadtxt('../../dataset/labels.csv', delimiter=',', skiprows=1)
y = np.argmax(y, axis=1)
X = np.where(np.isposinf(X), np.nanmax(X[np.isfinite(X)]), X)
X = np.where(np.isneginf(X), np.nanmin(X[np.isfinite(X)]), X)
feature_names = np.loadtxt('../../dataset/data.csv', delimiter=',', skiprows=0, dtype=str, max_rows=1)
feature_names = [feature.replace('_', ' ').capitalize() for feature in list(feature_names)]
feature_names = [feature.replace(' mean', '') for feature in feature_names]
target_names = np.loadtxt('../../dataset/labels.csv', delimiter=',', skiprows=0, dtype=str, max_rows=1)
target_names = [target.replace('_', ' ').capitalize() for target in list(target_names)]

xgb_estimator = xgb.XGBClassifier(max_depth=2,eta= 0.3,objective='multi:softmax',num_class=4,importance_type='weight')#,eval_metric='auc')
xgb_estimator.fit(X, y,verbose=True)

cv = 5  # Number of cross-validation folds
scores = cross_val_score(xgb_estimator, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Mean Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

# Classification report
y_pred = xgb_estimator.predict(X)
class_report = classification_report(y, y_pred, target_names=None)  # Optionally, provide target_names
output_path = '../../results/xgboost/classification_report.txt'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    f.write("Classification Report:\n")
    f.write(class_report)
    
# Confusion matrix
conf_matrix = confusion_matrix(y, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.matshow(conf_matrix, cmap='Greys')  # Using 'Greys' for the confusion matrix
plt.colorbar(cax)
ax.set_xlabel('Predicted Label', color='white')
ax.set_ylabel('True Label', color='white')
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(j, i, conf_matrix[i, j], ha='center', va='center', color='black' if conf_matrix[i, j] < conf_matrix.max() / 2 else 'white')
target_names = [f'Class {i}' for i in range(conf_matrix.shape[0])]  # Modify this if you have actual target names
ax.set_xticks(np.arange(len(target_names)))
ax.set_yticks(np.arange(len(target_names)))
ax.set_xticklabels(target_names, color='white')
ax.set_yticklabels(target_names, color='white')
plt.xticks(rotation=90, ha='right')
plt.tight_layout()
output_path = '../../results/xgboost/confusion_matrix.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

# Plot importance
importance = xgb_estimator.get_booster().get_score(importance_type='weight')
importance_dict = {feature_names[int(f[1:])]: score for f, score in importance.items()}
importance_df = pd.DataFrame(list(importance_dict.items()), columns=['Feature', 'Importance'])
importance_df = importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='maroon')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.yticks(fontsize=6)  # Set the font size of the y-axis tick labels (feature names)
plt.title('XGBoost Feature Importance')
plt.gca().invert_yaxis()  # Invert the y-axis to show the most important feature at the top
plt.tight_layout()
plt.savefig('../../results/xgboost/feature_importance.png')
plt.close()

# Top 10 plot
importance_df = importance_df.head(10)
plt.figure(figsize=(10, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='maroon')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 10 XGBoost Feature Importance')
plt.gca().invert_yaxis()  # Invert the y-axis to show the most important feature at the top
plt.tight_layout()
plt.savefig('../../results/xgboost/top10_feature_importance.png')
plt.close()