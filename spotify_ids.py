import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import base64
import json
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import urllib.parse
import os
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
import Levenshtein
import time

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer and stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocess_title(title):
    # Tokenize the title into words
    words = nltk.word_tokenize(title)
    # Lemmatize and stem each word
    processed_words = [stemmer.stem(lemmatizer.lemmatize(word.lower())) for word in words]
    # Join processed words back into a single string
    processed_title = ' '.join(processed_words)
    return processed_title

def is_approximately_same(title1, title2, threshold=0.8):
    processed_title1 = preprocess_title(title1)
    processed_title2 = preprocess_title(title2)
    distance = Levenshtein.distance(processed_title1, processed_title2)
    max_length = max(len(processed_title1), len(processed_title2))
    similarity = 1 - (distance / max_length)
    return similarity >= threshold

tags_to_genres = {
    "deep progressive house": " progressive house",
    "dark progressive house": " progressive house",
    "classic progressive house": " progressive house",
    "progressive house": " progressive house",
    "minimal dubstep": "dubstep",
    "deep dubstep": "dubstep",
    "dubstep": "dubstep",
    "filthstep": "dubstep",
    "brostep": "dubstep",
    "goa trance": "psytrance",
    "goa psytrance": "psytrance",
    "dark psytrance": "psytrance",
    "deep darkpsy": "psytrance",
    "forest psy": "psytrance",
    "deep minimal techno": "dark minimal techno",
    "dark techno": "dark minimal techno",
    "dark minimal techno": "dark minimal techno",
    "deep techno": "dark minimal techno",
    "minimal techno": "dark minimal techno",
    "german dark minimal techno": "dark minimal techno",
    'minimal dub': 'dark minimal techno',
}

genres = list(set(tags_to_genres.values()))
progressive_house_tags = ['progressive house','classic progressive house','deep progressive house','dark progressive house']
psytrance_tags = ['forest psy','goa trance', 'goa psytrance','deep darkpsy','dark psytrance']
dark_minimal_techno_tags = ['dark techno','dark minimal techno','german dark minimal techno','minimal dub']
dubstep_tags = ['dubstep','deep dubstep','filthstep','brostep']
tags = progressive_house_tags + psytrance_tags + dark_minimal_techno_tags + dubstep_tags
counter = {}
if os.path.exists('./results/counter.json'):
    with open('./results/counter.json') as f:
        counter = json.load(f)
if os.path.exists('./resutls/count.txt'):
    with open('./results/count.txt', 'r') as f:
        count = int(f.read())
else:
    counter = {key: 0 for key in tags}
    count = 0

with open('config.json') as f:
    config = json.load(f)
   
def save():
    filename = "results"

    if not os.path.exists(filename):
        os.mkdir(filename)

    zippedList =  list(zip(TracksIndexes, TracksName, TracksIds, TracksArtist, TracksArtistId, TracksTag, TracksGenre, TracksPreview, TracksAlbum, TracksAlbumId, TracksReleaseDate, TracksDuration, TracksPopularity))

    df = pd.DataFrame(zippedList, columns = ['Index', 'Name' , 'ID', 'Artist', "Artist's ID", 'Tag', 'Genre',  'Preview', 'Album', "Album's ID", "Release Date", "Duration", "Popularity"], index=TracksIndexes) 
    
    df.to_csv(f'{filename}/tracks.csv',index=False)
    send_email("Script Completed ", "Your Python script has finished successfully.")

    with open('./results/counter.json', 'w') as f:
        json.dump(counter, f)
    with open('./results/count.txt', 'w') as f:
        f.write(str(count))

def send_email(subject, body):
    # Email configuration
    sender_email = 'vlyberatos@gmail.com'  # Replace with your email address
    receiver_email = 'lyberatosbill@gmail.com'  # Replace with the recipient's email address
    password = config['password']  # Replace with your email password

    # Create the MIME object
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = receiver_email
    message['Subject'] = subject

    # Attach the body of the email
    message.attach(MIMEText(body, 'plain'))

    # Connect to the SMTP server (for Gmail, use 'smtp.gmail.com')
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()  # Start TLS encryption
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message.as_string())

def get_token():
    client_token = base64.b64encode("{}:{}".format(client_id, client_secret).encode('UTF-8')).decode('ascii')
    headers = {"Authorization": "Basic {}".format(client_token)}
    payload = {"grant_type": "client_credentials"}
    token_request = requests.post(SPOTIFY_TOKEN_URL, data=payload, headers=headers)
    access_token = json.loads(token_request.text)["access_token"]
    return access_token

def request_valid_song(access_token, genre=None, offset=0):

    # Make a request for the Search API with pattern and random index
    authorization_header = {"Authorization": "Bearer {}".format(access_token)}
    
    try:
        excluded_genres = [gen for gen in genres if gen != tags_to_genres[genre]]

        excluded_tags = [gen for gen in tags if tags_to_genres[gen] != tags_to_genres[genre]]
        excluded = excluded_genres + excluded_tags
        # Construct the query string to include one genre and exclude some genres
        included_genre_query = "genre:\"{}\"".format(genre)
        
        # Make the request to Spotify API with the query
        genre_query = urllib.parse.quote_plus(included_genre_query)
        song_request = requests.get(
                    '{}/search?q={}&type=track&offset={}'.format(
                        SPOTIFY_API_URL,
                        genre_query,
                        offset
                    ),
                    headers=authorization_header
                    )
        artist = list()
        artist_id = list()
        album = list()
        album_id = list()
        release_date = list()
        duration = list()
        popularity = list()
        preview_url = list()
        song = list()
        id = list()
        count_same = 0
        songs_info =[0]
        songs_info = json.loads(song_request.text)['tracks']['items']

        for song_info in songs_info:
            id_temp = song_info['id']
            if id_temp in TracksIds:
                count_same += 1
                continue
            if song_info['preview_url'] is None:
                continue
            else:
                                
                for names in TracksName:
                    if is_approximately_same(names,song_info['name']):
                        continue
                for genr in excluded:
                    if genr in song_info['name'] or genr in song_info['album']['name']:
                        continue

                id.append(song_info['id'])
                artist.append(song_info['artists'][0]['name'])
                artist_id.append(song_info['artists'][0]['id'])
                album.append(song_info['album']['name'])
                album_id.append(song_info['album']['id'])
                song.append(song_info['name'])
                popularity.append(song_info['popularity'])
                release_date.append(song_info['album']['release_date'])
                duration.append(song_info['duration_ms']/1000)
                preview_url.append(song_info['preview_url'])
                continue
    except IndexError as e:
        print("IndexError",e)
    except KeyError as e:
        print("KeyError",song_request.status_code)
        song.append("Error")
        artist.append(None)
        id.append(None)
        preview_url.append(None)
        artist_id.append(None)
        album.append(None)
        album_id.append(None)
        release_date.append(None)
        duration.append(None)
        popularity.append(None)
    except json.JSONDecodeError as e:
        song.append("Error")
        artist.append(None)
        id.append(None)
        preview_url.append(None)
        artist_id.append(None)
        album.append(None)
        album_id.append(None)
        release_date.append(None)
        duration.append(None)
        popularity.append(None)
        print("Too many requests, waiting for 10 seconds")
        time.sleep(10)        
    if count_same == len(songs_info):
        song.append("Same")
        artist.append(None)
        id.append(None)
        preview_url.append(None)
        artist_id.append(None)
        album.append(None)
        album_id.append(None)
        release_date.append(None)
        duration.append(None)
        popularity.append(None)
    if len(song)==0:
        song.append(None)
        artist.append(None)
        song.append(None)
        id.append(None)
        preview_url.append(None)
        artist_id.append(None)
        album.append(None)
        album_id.append(None)
        release_date.append(None)
        duration.append(None)
        popularity.append(None)
              
    return song,artist,artist_id,id,preview_url,album,album_id,release_date,duration,popularity

if __name__ == "__main__":
    try:

        # Extract client_id and client_secret
        client_id = config['client_id']
        client_secret = config['client_secret']
        client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager, requests_timeout = None) 

        # Spotify API URIs
        SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
        SPOTIFY_API_BASE_URL = "https://api.spotify.com"
        API_VERSION = "v1"
        SPOTIFY_API_URL = "{}/{}".format(SPOTIFY_API_BASE_URL, API_VERSION)

        TracksIndexes = list()
        TracksName = list()
        TracksIds = list()
        TracksArtist = list()
        TracksArtistId = list()
        TracksTag = list()
        TracksGenre = list()
        TracksPreview = list()
        TracksAlbum = list()
        TracksAlbumId = list()
        TracksReleaseDate = list()
        TracksDuration = list()
        TracksPopularity = list()

        file_path = './results/tracks.csv'
        if not os.path.exists(file_path):
            print(f"The file '{file_path}' does not exist.")
        else:
            df = pd.read_csv(file_path)

            TracksIndexes = df['Index'].tolist()
            TracksName = df['Name'].tolist()
            TracksIds = df['ID'].tolist()
            TracksArtist = df['Artist'].tolist()
            TracksArtistId = df["Artist's ID"].tolist()
            TracksTag = df['Tag'].tolist()
            TracksGenre = df['Genre'].tolist()
            TracksPreview = df['Preview'].tolist()
            TracksAlbum = df['Album'].tolist()
            TracksAlbumId = df["Album's ID"].tolist()
            TracksReleaseDate = df["Release Date"].tolist()
            TracksDuration = df["Duration"].tolist()
            TracksPopularity = df["Popularity"].tolist()

        # Get a Spotify API token
        access_token = get_token()  
        consecutive_none_count = 0
        number_of_genres = len(tags)
        while(True):
            for selected_genre in tags:
                song,artist,artist_id,id,preview_url,album,album_id,release_date,duration,popularity = request_valid_song(access_token, genre=selected_genre,offset=count)
                p = [numb for numb in range(len(TracksIds),len(TracksIds)+len(id))]

                if(song[0]=="Error"):
                    continue
                if(song[0]=="Same"):
                    counter[selected_genre] = counter[selected_genre] + 20
                elif(song[0]==None):
                    tags.remove(selected_genre)  # Remove genre from list
                    consecutive_none_count += 1  # Increase the count of consecutive None results
                else:
                    TracksIds.extend(id)
                    TracksName.extend(song)
                    TracksArtist.extend(artist)
                    TracksGenre.extend([tags_to_genres[selected_genre]]*len(id))
                    TracksTag.extend([selected_genre]*len(id))
                    TracksPreview.extend(preview_url)
                    TracksArtistId.extend(artist_id)
                    TracksAlbum.extend(album)
                    TracksAlbumId.extend(album_id)
                    TracksReleaseDate.extend(release_date)
                    TracksDuration.extend(duration)
                    TracksPopularity.extend(popularity)
                    TracksIndexes.extend(p)  
                    counter[selected_genre] = counter[selected_genre] + len(id)
                count = count + 20
            if(consecutive_none_count == number_of_genres):
                break
        save()
    except KeyboardInterrupt:  
        save()             
    except Exception as e:
        send_email("Script Error", f"An error occurred in your Python script:\n\n{str(e)}")
