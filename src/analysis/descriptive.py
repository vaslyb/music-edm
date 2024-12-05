import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, norm, boxcox, shapiro
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.covariance import EllipticEnvelope
import pandas as pd
import warnings
import argparse
import csv
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
            plt.close()

def generate_qqplots(genres_stats, output_dir):
    for feature in genres_stats[next(iter(genres_stats))]:
        print(f'Generating Q-Q plot for {feature}')
        if 'values' in genres_stats[next(iter(genres_stats))][feature]:
            plt.figure()
            data = [genres_stats[genre][feature]['values'] for genre in genres_stats]
            genres_stats_labels = [genre.replace('_', ' ').capitalize() for genre in genres_stats.keys()] 
            for i, feature_values in enumerate(data):
                norm_qq = norm.ppf(np.linspace(0.01, 0.99, len(feature_values)))
                plt.scatter(norm_qq, sorted(feature_values), label=genres_stats_labels[i], alpha=0.6)
            plt.title(f'Q-Q plot for {feature.replace("_", " ").capitalize()}')
            plt.xlabel('Theoretical Quantiles')
            plt.ylabel('Sample Quantiles')
            plt.legend()
            plt.savefig(os.path.join(output_dir, f'{feature}_qqplot.png'))
            plt.close()

def generate_histograms(genres_stats, output_dir):
    for feature in genres_stats[next(iter(genres_stats))]:
        print(f'Generating histogram for {feature}')
        if 'values' in genres_stats[next(iter(genres_stats))][feature]:
            plt.figure()
            
            # Filter out 'inf' values from the data
            data = [
                [v for v in genres_stats[genre][feature]['values'] if np.isfinite(v)] 
                for genre in genres_stats
            ]
            
            genres_stats_labels = [genre.replace('_', ' ').capitalize() for genre in genres_stats.keys()] 
            
            # Create histogram
            plt.hist(data, bins=30, stacked=True, label=genres_stats_labels, alpha=0.6)
            
            # Set the titles and labels
            plt.title(f'Histogram for {feature.replace("_", " ").capitalize()}')
            plt.xlabel(feature.replace("_", " ").capitalize())
            plt.ylabel('Frequency')
            
            # Add the legend
            plt.legend()
            
            # Save the plot
            plt.savefig(os.path.join(output_dir, f'{feature}_histogram.png'))
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

def perform_covariance_test(genres_stats, threshold=1e-2):
    """
    Perform covariance matrix equality test between genres using EllipticEnvelope.
    This tests the assumption of equal covariance matrices for Linear Discriminant Analysis.
    """
    genre_data = {}
    for genre, stats_data in genres_stats.items():
        genre_values = []
        for feature, data in stats_data.items():
            if 'values' in data:
                finite_values = [val for val in data['values'] if np.isfinite(val)]
                if len(finite_values) > 1:
                    genre_values.append(finite_values)
        if genre_values:
            genre_data[genre] = np.array(genre_values).T  # Transpose to align features properly
    
    # Perform the covariance test on the collected data
    cov_tests = {}
    genre_cov_matrices = {}  # To store the covariance matrices of each genre
    
    for genre, data in genre_data.items():
        envelope = EllipticEnvelope()
        try:
            envelope.fit(data.T)
            genre_cov_matrices[genre] = envelope.covariance_
            cov_tests[genre] = 'Covariance matrix calculated successfully'
        except Exception as e:
            cov_tests[genre] = str(e)
    
    # Compare covariance matrices between genres
    passed = True
    cov_matrix_list = list(genre_cov_matrices.values())
    for i in range(len(cov_matrix_list)):
        for j in range(i+1, len(cov_matrix_list)):
            cov_matrix1 = cov_matrix_list[i]
            cov_matrix2 = cov_matrix_list[j]
            
            # Calculate the Frobenius norm of the difference between the matrices
            frobenius_norm = np.linalg.norm(cov_matrix1 - cov_matrix2, 'fro')
            
            # Check if the difference is within the threshold
            if frobenius_norm > threshold:
                passed = False
                print(f"Covariance matrices for genres are significantly different. Frobenius norm: {frobenius_norm}")
    
    if passed:
        print("Covariance matrices are similar across genres. Test passed!")
    else:
        print("Covariance matrices are significantly different. Test failed!")
    
    return cov_tests

def shapiro_wilk_test(genres_stats, output_file, alpha=0.05):
    """
    Perform the Shapiro-Wilk test for normality per feature and genre.
    Saves the results in a CSV file, including the final decision (whether it passes the normality test).
    """
    # Create a list to store the results
    results = []

    for genre, stats_data in genres_stats.items():
        for feature, data in stats_data.items():
            if 'values' in data:
                values = np.array(data['values'])
                # Perform Shapiro-Wilk test for normality
                stat, p_value = shapiro(values)
                # Determine if the feature passes the normality test
                decision = 'Pass' if p_value > alpha else 'Fail'
                
                # Append the results for this genre, feature, and normality test decision
                results.append({
                    'Genre': genre,
                    'Feature': feature,
                    'Statistic': stat,
                    'P-value': p_value,
                    'Decision': decision
                })
    
    # Write the results to a CSV file
    with open(output_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Genre', 'Feature', 'Statistic', 'P-value', 'Decision'])
        writer.writeheader()
        writer.writerows(results)

    print(f"Shapiro-Wilk test results saved to {output_file}")

def main(root_dir, output_dir):
    genres_stats = {}
    for genre in os.listdir(root_dir):
        genre_dir = os.path.join(root_dir, genre)
        if os.path.isdir(genre_dir):
            genres_stats[genre] = process_genre_directory(genre_dir)
    
    # Generate boxplots, QQ plots, histograms
    generate_boxplots(genres_stats, output_dir)
    generate_qqplots(genres_stats, output_dir)
    generate_histograms(genres_stats, output_dir)
    
    # Perform T-tests
    t_test_results = perform_t_tests(genres_stats)
    t_test_results.to_csv(os.path.join(output_dir, 't_test_results.csv'), index=False)
    
    # Perform covariance tests
    cov_tests = perform_covariance_test(genres_stats)
    with open(os.path.join(output_dir, 'cov_tests.json'), 'w') as json_file:
        json.dump(cov_tests, json_file)
    
    # Perform Shapiro-Wilk tests
    shapiro_results = shapiro_wilk_test(genres_stats, output_file=os.path.join(output_dir, 'shapiro_wilk_results.csv'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Music Genre Feature Analysis")
    parser.add_argument('--input_dir', type=str, help="Root directory containing genre folders")
    parser.add_argument('--output_dir', type=str, help="Directory to save output files")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # Call the main function with parsed arguments
    main(args.input_dir, args.output_dir)
