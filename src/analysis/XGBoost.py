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
import argparse
import warnings

warnings.filterwarnings("ignore")

# Set up argument parsing
parser = argparse.ArgumentParser(description='DT Model Script')
parser.add_argument('--input_dir', type=str, required=True, help='Directory for input data files')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results')
parser.add_argument('--select_features', action='store_true', help='Whether to select specific features or use all')

args = parser.parse_args()

save_path = args.output_dir
input_path = args.input_dir
os.makedirs(save_path, exist_ok=True)

# Load the dataset
X = np.loadtxt(f'{input_path}/data.csv', delimiter=',', skiprows=1, usecols=range(1, np.genfromtxt(f'{input_path}/data.csv', delimiter=',', max_rows=1).size))
y = np.loadtxt(f'{input_path}/labels.csv', delimiter=',', skiprows=1, usecols=range(1, np.genfromtxt(f'{input_path}/labels.csv', delimiter=',', max_rows=1).size))
y = np.argmax(y, axis=1)

# Feauture Selection
feature_names = np.loadtxt(f'{input_path}/data.csv', delimiter=',', skiprows=0, dtype=str, max_rows=1)
feature_names = feature_names[1:]

target_names = np.loadtxt(f'{input_path}/labels.csv', delimiter=',', skiprows=0, dtype=str, max_rows=1)
target_names = target_names[1:]
target_names = [target.replace('_', ' ').capitalize() for target in list(target_names)]

if args.select_features:
    disgard_features = ['spectral_energy_mean','pulse_clarity_mean','attack_slope_mean','spectral_flatness_mean','entropia_clarity','attack_time',
                        'spectral_flux_mean','danceability','chroma1_mean','chroma2_mean','chroma3_mean','chroma4_mean','chroma5_mean','chroma6_mean',
                        'chroma7_mean','chroma8_mean','chroma9_mean','chroma10_mean','chroma11_mean','chroma12_mean','mfcc6_mean','mfcc7_mean',
                        'mfcc8_mean','mfcc9_mean','mfcc10_mean','mfcc11_mean','mfcc12_mean','mfcc13_mean','chord','chord_strength','chord_scale'
                        ,'key','key_strength','spectral_rms_mean', 'meter','entropia_clarity','entropia_clarity_low','entropia_clarity_high',
                        'entropia_clarity_middle']
    features_to_keep_index = [index for index, feature in enumerate(feature_names) if feature not in disgard_features]
    features_to_keep = [feature for index, feature in enumerate(feature_names) if feature not in disgard_features]
    features_to_keep = [feature.replace('_', ' ').capitalize() for feature in list(features_to_keep)]
    features_to_keep = [feature.replace(' mean', '') for feature in features_to_keep]
else:
    features_to_keep_index = range(len(feature_names))
    features_to_keep = feature_names
    features_to_keep = [feature.replace('_', ' ').capitalize() for feature in list(features_to_keep)]
    features_to_keep = [feature.replace(' mean', '') for feature in features_to_keep]
print(features_to_keep)
X = X[:,features_to_keep_index]

xgb_estimator = xgb.XGBClassifier(max_depth=2,eta= 0.3,objective='multi:softmax',num_class=4,importance_type='weight')
xgb_estimator.fit(X, y,verbose=True)

cv = 5  # Number of cross-validation folds
scores = cross_val_score(xgb_estimator, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Mean Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

# Classification report
y_pred = xgb_estimator.predict(X)
class_report = classification_report(y, y_pred, target_names=None)  # Optionally, provide target_names
output_path = f'{save_path}/classification_report.txt'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    f.write("Classification Report:\n")
    f.write(class_report)
    
# Confusion matrix
conf_matrix = confusion_matrix(y, y_pred)
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
conf_matrix_percentage = np.round(conf_matrix_normalized).astype(int)
fig, ax = plt.subplots(figsize=(10, 8))  # Change the proportions here
cax = ax.matshow(conf_matrix_percentage, cmap='Greys', aspect='auto')  # Set aspect to auto for better proportioning
plt.colorbar(cax)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
for i in range(len(target_names)):
    for j in range(len(target_names)):
        ax.text(j, i, f'{conf_matrix_percentage[i, j]:}%', ha='center', va='center', fontsize=16, color='black' if i!=j else 'white')
ax.set_xticks(np.arange(len(target_names)))
ax.set_yticks(np.arange(len(target_names)))
ax.set_xticklabels(target_names,fontsize=14)
ax.set_yticklabels(target_names,fontsize=14)
plt.xticks(rotation=90, ha='right')
plt.tight_layout()
plt.savefig(f'{save_path}/confusion_matrix.png')  # Save as PNG file
plt.close()

# Plot importance
importance = xgb_estimator.get_booster().get_score(importance_type='weight')
importance_dict = {features_to_keep[int(f[1:])]: score for f, score in importance.items()}
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
plt.savefig(f'{save_path}/feature_importance.png')
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
plt.savefig(f'{save_path}/top10_feature_importance.png')
plt.close()