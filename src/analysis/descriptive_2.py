import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Load the JSON file
with open('./results/genre_statistics.json', 'r') as file:
    data = json.load(file)

# Prepare lists to store mean and variance values
means = {}
variances = {}

# Extract mean and variance data
for genre, features in data.items():
    for feature, stats in features.items():
        if feature not in means:
            means[feature] = []
        if feature not in variances:
            variances[feature] = []
        means[feature].append(stats['mean'])
        variances[feature].append(stats['variance'])

# Create DataFrame from extracted data
df_mean = pd.DataFrame(means, index=data.keys())
df_variance = pd.DataFrame(variances, index=data.keys())

# Set up styles for plots
plt.style.use('fast')  # Choose a style that suits your preference

# Generate and save visualizations for each feature's mean and variance
for feature in df_mean.columns:
    plt.figure(figsize=(10, 6))
    cleaned_index = [item.replace('_', ' ') for item in df_mean.index]

    # Convert the cleaned list back to a Pandas Index
    cleaned_index = pd.Index(cleaned_index, dtype='object')
    # Plot mean values with error bars for variance
    plt.bar(cleaned_index, df_mean[feature], yerr=np.sqrt(df_variance[feature]), capsize=5, color='skyblue', edgecolor='gray')  # Adjust colors and edgecolor

    # Customize plot appearance
    plt.title(f'{feature.replace("_mean", "").replace("_"," ")} ', fontsize=16)  # Improve title
    plt.xlabel('Genre', fontsize=12)  # Label font size
    plt.ylabel('Value', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate x-axis labels and adjust fontsize
    plt.yticks(fontsize=10)  # Adjust y-axis label fontsize
    plt.ylim(bottom=0)  # Ensure y-axis starts at 0

    # Add grid lines and remove unnecessary spines
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()

    # Save the figure with a more descriptive filename
    output_dir = './results/visualizations/'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}{feature.replace("_mean", "").lower()}.png')

    plt.close()


# Set up styles for plots
plt.style.use('fast')  # Choose a style that suits your preference

# Filter features containing 'chroma' and 'mfcc'
feature_chroma = [feat for feat in df_mean.columns if 'chroma' in feat]
feature_mfcc = [feat for feat in df_mean.columns if 'mfcc' in feat]

# Generate a colormap for genres
genres = df_mean.index
colors = plt.cm.viridis(np.linspace(0, 1, len(genres)))
genre_colors = dict(zip(genres, colors))

# Function to plot features for each genre with assigned colors
def plot_features_for_genre(genre, features, title, output_filename):
    plt.figure(figsize=(14, 8))
    
    # Extract data for the genre
    genre_mean = df_mean.loc[genre, features]
    genre_variance = df_variance.loc[genre, features]

    # Plot bar chart for the genre's features
    plt.bar(features, genre_mean, yerr=np.sqrt(genre_variance), capsize=5, color=genre_colors[genre], edgecolor='gray')
    
    # Customize plot appearance
    plt.title(f'{title} for {genre.replace("_", " ")}', fontsize=16)
    plt.xlabel(f'{title} Features', fontsize=12)
    plt.ylabel('Mean Value', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(bottom=0)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()

    # Save the figure with a descriptive filename
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    plt.savefig(output_filename)
    plt.close()

# Plot and save chroma features for each genre
for genre in df_mean.index:
    plot_features_for_genre(genre, feature_chroma, 'Chroma Features', f'./results/visualizations/chroma_per_genre/{genre.replace(" ", "_").lower()}_chroma_features.png')

# Plot and save mfcc features for each genre
for genre in df_mean.index:
    plot_features_for_genre(genre, feature_mfcc, 'MFCC Features', f'./results/visualizations/mfcc_per_genre/{genre.replace(" ", "_").lower()}_mfcc_features.png')
