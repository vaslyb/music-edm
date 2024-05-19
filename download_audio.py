import pandas as pd
import os
from spotify_ids import send_email


if __name__ == "__main__":
    try:
        csv_file = "./results/tracks.csv"

        # Read the CSV file
        df = pd.read_csv(csv_file)
        # Extract the 'preview_url' and 'ID' columns
        preview_urls = df["Preview"].values.tolist()
        IDs = df["ID"].values.tolist()
        Genres = df["Genre"].values.tolist()
        if not os.path.exists("./results/audio"):
            os.mkdir("./results/audio")
        for preview_url, ID, Genre in zip(preview_urls, IDs, Genres):
            Genre = Genre[1:] if Genre[0] == " " else Genre
            Genre = Genre.replace(" ", "_")
            if not os.path.exists("./results/audio/" + str(Genre)):
                os.mkdir("./results/audio/" + str(Genre))
            os.system(
                "wget -c "
                + preview_url
                + " -O "
                + "./results/audio/"
                + str(Genre)
                + "/"
                + str(ID)
                + ".mp3"
            )
    except Exception as e:
        send_email(
            "Script Error", f"An error occurred in your Python script:\n\n{str(e)}"
        )
        print(f"An error occurred in your Python script:\n\n{str(e)}")
