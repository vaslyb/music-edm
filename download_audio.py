import pandas as pd
import os

csv_file = 'tracks.csv'

# Read the CSV file
df = pd.read_csv(csv_file)

# Extract the 'preview_url' and 'ID' columns
preview_urls = df["Preview"].values.tolist()
IDs = df["ID"].values.tolist()

# Download audio files
for preview_url, ID in zip(preview_urls, IDs):
    os.system('wget -c ' + preview_url + ' -O ' + "./audio/" + str(ID) + '.mp3')
