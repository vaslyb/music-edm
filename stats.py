import os
import json
import numpy as np

# Dictionary mapping musical notes to numerical values
note_mapping = {
    'C': 1, 'C#': 2, 'D': 3, 'D#': 4, 'E': 5, 'F': 6, 
    'F#': 7, 'G': 8, 'G#': 9, 'A': 10, 'A#': 11, 'B': 12
}

def translate_to_numeric(value):
    semitone_offset = note_mapping[value]
    return semitone_offset

def process_genre_directory(genre_dir):
    """
    Process all JSON files in a genre directory and calculate statistics.
    Translate 'chord scale' and 'key scale' to numerical values.
    """
    feature_sums = {}
    feature_counts = {}
    feature_sumsq = {}
    
    for filename in os.listdir(genre_dir):
        if filename.endswith(".json"):
            with open(os.path.join(genre_dir, filename), 'r') as f:
                track_data = json.load(f)
                for feature, values in track_data.items():
                    if feature == 'key' or feature == 'chord':
                        translated_value = translate_to_numeric(values)
                        values = float(translated_value)
                    if feature == 'key_scale' or feature == 'chord_scale':
                        values = np.where(values == 'major', 1, values)
                        values = np.where(values == 'minor', 0, values)
                        values = float(values)    
                    if feature not in feature_sums:
                        feature_sums[feature] = np.zeros_like(values, dtype=np.float64)
                        feature_counts[feature] = 0
                        feature_sumsq[feature] = np.zeros_like(values, dtype=np.float64)
                    feature_sums[feature] += values
                    feature_sumsq[feature] += values ** 2
                    feature_counts[feature] += 1
                        
    genre_stats = {}
    for feature in feature_sums:
        mean = np.divide(feature_sums[feature], feature_counts[feature])
        mean_squares = np.divide(feature_sumsq[feature], feature_counts[feature])
        variance = mean_squares - mean ** 2
        genre_stats[feature] = {'mean': mean.tolist(), 'variance': variance.tolist()}  # Convert numpy arrays to lists
    
    return genre_stats

def main(root_dir):
    genres_stats = {}
    for genre in os.listdir(root_dir):
        genre_dir = os.path.join(root_dir, genre)
        if os.path.isdir(genre_dir):
            genres_stats[genre] = process_genre_directory(genre_dir)
    
    return genres_stats

if __name__ == "__main__":
    root_directory = "./results/features/"
    genre_statistics = main(root_directory)
    with open("./results/genre_statistics.json", "w") as outfile:
        json.dump(genre_statistics, outfile, indent=4)
