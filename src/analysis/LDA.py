import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import os 
import pandas as pd
import seaborn as sns
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description='LDA Model Script')
parser.add_argument('--input_dir', type=str, required=True, help='Directory for input data files')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results')
parser.add_argument('--select_features', action='store_true', help='Whether to select specific features or use all')

args = parser.parse_args()

# Define model
model = LinearDiscriminantAnalysis()

# path to save the resutls
save_path = args.output_dir
input_path = args.input_dir

# Define model evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# features and labels
feature_names = np.loadtxt(f'{input_path}/data.csv', delimiter=',', skiprows=0, dtype=str, max_rows=1)
feature_names = feature_names[1:]
target_names = np.loadtxt(f'{input_path}/labels.csv', delimiter=',', skiprows=0, dtype=str, max_rows=1)
target_names = target_names[1:]
num_features = len(feature_names)
num_labels = len(target_names)

# Load the dataset
X = np.loadtxt(f'{input_path}/data.csv', delimiter=',', skiprows=1, usecols=range(1, num_features+1))
y = np.loadtxt(f'{input_path}/labels.csv', delimiter=',', skiprows=1, usecols=range(1, num_labels+1))

# Feauture Selection
target_names = [target.replace('_', ' ').capitalize() for target in list(target_names)]
disgard_features = ['spectral_energy_mean','pulse_clarity_mean','attack_slope_mean','spectral_flatness_mean','entropia_clarity','attack_time',
                    'spectral_flux_mean','danceability','chroma1_mean','chroma2_mean','chroma3_mean','chroma4_mean','chroma5_mean','chroma6_mean',
                    'chroma7_mean','chroma8_mean','chroma9_mean','chroma10_mean','chroma11_mean','chroma12_mean','mfcc6_mean','mfcc7_mean',
                    'mfcc8_mean','mfcc9_mean','mfcc10_mean','mfcc11_mean','mfcc12_mean','mfcc13_mean','chord','chord_strength','chord_scale'
                    ,'key','key_strength','spectral_rms_mean', 'meter','entropia_clarity','entropia_clarity_low','entropia_clarity_high',
                    'entropia_clarity_middle']
if args.select_features:
    features_to_keep_index = [index for index, feature in enumerate(feature_names) if feature not in disgard_features]
    features_to_keep = [feature for index, feature in enumerate(feature_names) if feature not in disgard_features]
    features_to_keep = [feature.replace('_', ' ').capitalize() for feature in list(features_to_keep)]
    features_to_keep = [feature.replace(' mean', '') for feature in features_to_keep]
else:
    features_to_keep_index = range(num_features)
    features_to_keep = feature_names
    features_to_keep = [feature.replace('_', ' ').capitalize() for feature in list(features_to_keep)]
    features_to_keep = [feature.replace(' mean', '') for feature in features_to_keep]
X = X[:,features_to_keep_index]

# Standardize the data
standarizer = StandardScaler()
X = standarizer.fit_transform(X)

# Transform one hot encoded labels to integers
y = np.argmax(y, axis=1)

# Fit model
model.fit(X, y)

# Predict class labels
y_pred = model.predict(X)

# Predict class probabilities
y_prob = model.predict_proba(X)

# Evaluate Model with Cross-Validation
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Mean Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

# Interpretations
os.makedirs(save_path, exist_ok=True)


# Explained variance (captures the ratio of the total variance each principal component captures, how whel each LD separates the classes)
print("Explained Variance Ratio:")
print(model.explained_variance_ratio_)

# Coefficients (In LDA, the coefficients are weights associated with each feature for constructing the Linear Discriminants. Each Linear Discriminant is a linear combination of the original features, and the coefficients determine the contribution of each feature to this combination.)
coefficients = model.coef_
coefficients_df = pd.DataFrame(coefficients, columns=features_to_keep, index=target_names)
coefficients_df.to_csv(save_path+'/coefficients.csv')

# Confusion Matrix
conf_matrix = confusion_matrix(y, y_pred)
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
conf_matrix_percentage = np.round(conf_matrix_normalized).astype(int)
fig, ax = plt.subplots(figsize=(10, 8))  # Change the proportions here
cax = ax.matshow(conf_matrix_percentage, cmap='Reds', aspect='auto')  # Set aspect to auto for better proportioning
plt.colorbar(cax)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
for i in range(len(target_names)):
    for j in range(len(target_names)):
        ax.text(j, i, f'{conf_matrix_percentage[i, j]:}%', ha='center', va='center', color='black' if conf_matrix[i, j] < conf_matrix.max() / 2 else 'white', fontsize=16)
ax.set_xticks(np.arange(len(target_names)))
ax.set_yticks(np.arange(len(target_names)))
ax.set_xticklabels(target_names,fontsize=14)
ax.set_yticklabels(target_names,fontsize=14)
plt.xticks(rotation=90, ha='right')
plt.tight_layout()
plt.savefig(f'{save_path}/confusion_matrix.png')  # Save as PNG file
plt.close()

# Classification report
class_report = classification_report(y, y_pred)
print("Classification Report:")
print(class_report)
with open(f'{save_path}/classification_report.txt', 'w') as f:
    f.write(class_report)

# Plotting the LDs
X_r = model.transform(X)
n_lds = X_r.shape[1]
colors = ['red', 'green', 'blue', 'purple']
target_names = [target.replace('_', ' ').capitalize() for target in list(target_names)]
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
        plt.savefig(f'{save_path}/ld{i + 1}_vs_ld{j + 1}.png')


abs_df = coefficients_df.abs()

# Find the top 10 columns based on the maximum absolute value in each column
top_columns = abs_df.max().nlargest(10).index

# Keep only those columns in the original DataFrame
filtered_df = coefficients_df[top_columns]

filtered_df_transposed = filtered_df.T

# Set the aesthetics for the plots
plt.figure(figsize=(15, 10))

# Heatmap with inverted axes
sns.heatmap(filtered_df, annot=True, cmap='coolwarm', cbar=True)
plt.title('Heatmap of Top 10 Features (Inverted Axes)')
plt.xlabel('Samples')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig(f'{save_path}/heatmap.png')
plt.close()

# Bar Plot
plt.figure(figsize=(10, 6))
filtered_df_transposed.plot(kind='bar')
plt.title('Bar Plot of Top 10 Features')
plt.xlabel('Features')
plt.ylabel('Values')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.legend(title='Features', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f'{save_path}/bar_plot.png')
plt.close()
