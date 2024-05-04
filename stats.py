import pandas as pd

# Read the CSV file
df = pd.read_csv('./results/tracks.csv')

# Calculate the number of instances of each class

genre = df['Genre'].value_counts()

# Create a DataFrame for the statistics

subgenre_stats_df = pd.DataFrame({'Genre': genre.index, 'Count': genre.values})
# Save the statistics to a new CSV file

subgenre_stats_df.to_csv('./results/genre_stats.csv', index=False)
