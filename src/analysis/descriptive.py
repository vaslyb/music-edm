import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import pandas as pd

# Define the note_mapping dictionary for translate_to_numeric function
note_mapping = {
    'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
}

def translate_to_numeric(value):
    semitone_offset = note_mapping[value]
    return semitone_offset

def process_genre_directory(genre_dir):
    """
    Process all JSON files in a genre directory and calculate statistics.
    Translate 'chord scale' and 'key scale' to numerical values.
    """
    feature_values = {}
    
    for filename in os.listdir(genre_dir):
        if filename.endswith(".json"):
            with open(os.path.join(genre_dir, filename), 'r') as f:
                track_data = json.load(f)
                for feature, values in track_data.items():
                    if feature == 'key' or feature == 'chord':
                        translated_value = translate_to_numeric(values)
                        values = float(translated_value)
                    elif feature == 'key_scale' or feature == 'chord_scale':
                        values = 1.0 if values == 'major' else 0.0
                    
                    if feature not in feature_values:
                        feature_values[feature] = []
                    feature_values[feature].append(values)
                        
    genre_stats = {}
    for feature in feature_values:
        values = np.array(feature_values[feature], dtype=np.float64)
        mean = np.mean(values)
        variance = np.var(values)
        genre_stats[feature] = {'mean': mean, 'variance': variance, 'values': values.tolist()}
    
    return genre_stats

def generate_boxplots(genres_stats, output_dir):
    for feature in genres_stats[next(iter(genres_stats))]:
        print(f'Generating boxplot for {feature}')
        if 'values' in genres_stats[next(iter(genres_stats))][feature]:
            plt.figure()
            data = [genres_stats[genre][feature]['values'] for genre in genres_stats]
            genres_stats_labels = [genre.replace('_', ' ').capitalize() for genre in genres_stats.keys()] 
            plt.boxplot(data, labels=genres_stats_labels, patch_artist=False, showmeans=True, meanline=True, showfliers=True)
            plt.title(feature.replace('_', ' ').capitalize())
            plt.xlabel('Genre')
            plt.ylabel(feature.replace('_', ' ').capitalize())
            plt.savefig(os.path.join(output_dir, f'{feature}_boxplot.png'))
            plt.show()
            plt.close()

def perform_t_tests(genres_stats):
    t_test_results = []
    genres = list(genres_stats.keys())
    for i in range(len(genres)):
        for j in range(i+1, len(genres)):
            genre1 = genres[i]
            genre2 = genres[j]
            for feature in genres_stats[genre1]:
                if 'values' in genres_stats[genre1][feature]:
                    values1 = genres_stats[genre1][feature]['values']
                    values2 = genres_stats[genre2][feature]['values']
                    t_stat, p_value = ttest_ind(values1, values2, equal_var=False)
                    t_test_results.append({
                        'Feature': feature,
                        'Genre1': genre1,
                        'Genre2': genre2,
                        'T-statistic': t_stat,
                        'P-value': p_value
                    })
    
    # Create a DataFrame from the t-test results
    t_test_df = pd.DataFrame(t_test_results)
    return t_test_df
    
    return t_test_results

def main(root_dir, output_dir):
    genres_stats = {}
    for genre in os.listdir(root_dir):
        genre_dir = os.path.join(root_dir, genre)
        if os.path.isdir(genre_dir):
            genres_stats[genre] = process_genre_directory(genre_dir)
    
    # Generate boxplots
    generate_boxplots(genres_stats, output_dir)
    
    # Perform t-tests and get results as a DataFrame
    t_test_df = perform_t_tests(genres_stats)
    
    # Save statistics and t-test results
    with open(os.path.join(output_dir, "genre_statistics.json"), "w") as outfile:
        json.dump(genres_stats, outfile, indent=4)
    
    # Save the t-test results to a CSV file
    t_test_df.to_csv(os.path.join(output_dir, "t_test_results.csv"), index=False)
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    root_directory = "../../results/features/"
    output_directory = "../../results/statistics/"
    os.makedirs(output_directory, exist_ok=True)
    main(root_directory, output_directory)
