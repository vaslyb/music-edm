import pandas as pd

# Read the CSV file
df = pd.read_csv('tracks.csv')

# Calculate the number of instances of each class
class_counts = df['Main Genre'].value_counts()

# Create a DataFrame for the statistics
stats_df = pd.DataFrame({'Genre': class_counts.index, 'Count': class_counts.values})

# Save the statistics to a new CSV file
stats_df.to_csv('stats.csv', index=False)
