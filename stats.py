import pandas as pd

# Read the CSV file
df = pd.read_csv('tracks.csv')

# Calculate the number of instances of each class
genre = df['Main Genre'].value_counts()
subgenre = df['Genre'].value_counts()

# Create a DataFrame for the statistics
stats_df = pd.DataFrame({'Main Genre': genre.index, 'Count': genre.values})
subgenre_stats_df = pd.DataFrame({'Genre': subgenre.index, 'Count': subgenre.values})
# Save the statistics to a new CSV file
stats_df.to_csv('main_genre_stats.csv', index=False)
subgenre_stats_df.to_csv('genre_stats.csv', index=False)
