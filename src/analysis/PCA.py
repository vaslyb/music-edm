import numpy as np
import pandas as pd
import os
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import argparse

warnings.filterwarnings("ignore")

# Argument parser setup
parser = argparse.ArgumentParser(description='PCA analysis script with optional feature selection.')
parser.add_argument('--input_dir', type=str, required=True, help='Directory for input data files')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results')
parser.add_argument('--select_features', action='store_true', help='Whether to select specific features or use all')

args = parser.parse_args()

# Ensure the input directory exists
if not os.path.exists(args.input_dir):
    raise FileNotFoundError(f"Input directory {args.input_dir} does not exist.")

save_path = args.output_dir
input_path = args.input_dir
os.makedirs(save_path, exist_ok=True)

# Load the dataset
data_file = os.path.join(args.input_dir, 'data.csv')
labels_file = os.path.join(args.input_dir, 'labels.csv')

X = np.loadtxt(data_file, delimiter=',', skiprows=1, usecols=range(1, np.genfromtxt(data_file, delimiter=',', max_rows=1).size))
y = np.loadtxt(labels_file, delimiter=',', skiprows=1, usecols=range(1, np.genfromtxt(labels_file, delimiter=',', max_rows=1).size))

# Transform one hot encoded labels to integers
y = np.argmax(y, axis=1)

# Load feature and target names
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
                        ,'key','key_strength']
    # disgard_features_2 = ['chroma1_mean','chroma2_mean','chroma3_mean','chroma4_mean','chroma5_mean','chroma6_mean',
    #                       'chroma7_mean','chroma8_mean','chroma9_mean','chroma10_mean','chroma11_mean','chroma12_mean',
    #                       'spectral_energy_mean','pulse_clarity_mean','attack_slope_mean','spectral_flatness_mean',
    #                       'pulse_clarity_mean','attack_slope_mean','spectral_flatness_mean','entropia_clarity','attack_time',
    #                       'spectral_flux_mean','danceability']
    features_to_keep_index = [index for index, feature in enumerate(feature_names) if feature not in disgard_features]
    features_to_keep = [feature for index, feature in enumerate(feature_names) if feature not in disgard_features]
else:
    features_to_keep_index = range(len(feature_names))
    features_to_keep = feature_names


X = X[:,features_to_keep_index]
# Create dataframe from selected features
df = pd.read_csv(data_file)
df = df[features_to_keep]

# Replace infinite values in the dataframe
df.replace({np.inf: df[np.isfinite(df)].max().max(), -np.inf: df[np.isfinite(df)].min().min()}, inplace=True)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Apply PCA
pca = PCA()
principal_components = pca.fit_transform(X_scaled)

# Explained variance ratio and cumulative variance
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

# Plot and save cumulative explained variance
plt.plot(cumulative_variance)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance by PCA Components')
plt.grid()
plt.axhline(y=0.90, color='r', linestyle='--')  # Horizontal line at 90%
plt.savefig(os.path.join(args.output_dir, 'cumulative_variance.png'))
plt.close()

# Threshold for explained variance
threshold = 0.90
num_components = np.argmax(cumulative_variance >= threshold) + 1

# Apply PCA with the selected number of components
pca = PCA(n_components=num_components)
principal_components = pca.fit_transform(X_scaled)

# Save loadings
loadings = pd.DataFrame(pca.components_, columns=df.columns)
loadings.to_csv(os.path.join(save_path, 'loadings.csv'), index=False)

# Group significant features for each component
loading_threshold = 0.2
component_groups = {}

for i in range(loadings.shape[0]):
    component_name = f"Component {i + 1}"
    significant_features = loadings.iloc[i][
        loadings.iloc[i].abs() > loading_threshold
    ].index.tolist()
    component_groups[component_name] = significant_features

# Convert to a DataFrame and save the grouped features
grouped_features_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in component_groups.items()]))
grouped_features_df.to_csv(os.path.join(save_path, 'grouped_features.csv'), index=False)

print(f"PCA analysis complete. Results saved to {save_path}")
