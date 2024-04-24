import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import pickle
import base64
import json
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import urllib.parse
import time
import os

subgenres_to_genres = {
    "deep tech house": "house",
    "deep progressive house": "house",
    "dark progressive house": "house",
    "funky tech house": "house",
    "deep deep tech house": "house",
    "tech house": "house",
    "deep deep house": "house",
    "jackin’ house": "house",
    "funky house": "house",
    "organic house": "house",
    "minimal tech house": "house",
    "classic house": "house",
    "tribal house": "house",
    "deep euro house": "house",
    "soulful house": "house",
    "deep house": "house",
    "bouncy house": "house",
    "deep groove house": "house",
    "diva house": "house",
    "bass house": "house",
    "fidget house": "house",
    "float house": "house",
    "classic progressive house": "house",
    "progressive house": "house",
    "slap house": "house",
    "deep tropical house": "house",
    "tropical house": "house",
    "stutter house": "house",
    "future house": "house",
    "minimal dubstep": "dubstep",
    "deep dubstep": "dubstep",
    "deep filthstep": "dubstep",
    "classic dubstep": "dubstep",
    "filthstep": "dubstep",
    "gaming dubstep": "dubstep",
    "deathstep": "dubstep",
    "riddim dubstep": "dubstep",
    "tearout": "dubstep",
    "brostep": "dubstep",
    "goa trance": "trance",
    "goa psytrance": "trance",
    "progressive psytrance": "trance",
    "psychedelic trance": "trance",
    "dark psytrance": "trance",
    "tech trance": "trance",
    "cosmic uplifting trance": "trance",
    "bubble trance": "trance",
    "deep psytrance": "trance",
    "deep progressive trance": "trance",
    "progressive uplifting trance": "trance",
    "deep uplifting trance": "trance",
    "acid trance": "trance",
    "progressive trance": "trance",
    "old school hard trance": "trance",
    "forest psy": "trance",
    "deep full on": "trance",
    "full on": "trance",
    "deep minimal techno": "techno",
    "dark techno": "techno",
    "raw techno": "techno",
    "dark minimal techno": "techno",
    "deep techno": "techno",
    "minimal techno": "techno",
    "acid techno": "techno",
    "industrial techno": "techno",
    "modular techno": "techno",
    "bleep techno": "techno",
    "deep liquid bass": "drum and bass",
    "jump up": "drum and bass",
    "neuro step": "drum and bass",
    "darkstep": "drum and bass",
    "neurofunk": "drum and bass",
    "dark psytrance": "trance",
    "deep darkpsy": "trance",
    "german dark minimal techno": "techno",
    "dubstep": "dubstep",
    'minimal dub': 'techno',
}

genres = ['house', 'dubstep', 'trance', 'techno','drum and bass']

house_genres = ['progressive house','classic progressive house','deep progressive house','dark progressive house']
trance_genres = ['forest psy','goa trance', 'goa psytrance','deep darkpsy','dark psytrance']
techno_genres = ['dark techno','dark minimal techno','german dark minimal techno','minimal dub']
dubstep_genres = ['dubstep','deep dubstep','filthstep','brostep']
subgenres = house_genres + trance_genres + techno_genres + dubstep_genres 
subgenres_original = subgenres

# Open genres file
# try:
#        genre_file = 'genres.json'
#     with open(genre_file, 'r') as infile:
#         subgenres = json.load(infile)
#         subgenres_original = subgenres
# except FileNotFoundError:
#     print("Couldn't find genres file!")
#     sys.exit(1)

# Load credentials from config file
with open('config.json') as f:
    config = json.load(f)
    
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
    
    song = None
    for _ in range(2):
        try:
            excluded_genres = [gen for gen in genres if gen != subgenres_to_genres[genre]]

            excluded_subgenres = [gen for gen in subgenres if subgenres_to_genres[gen] != subgenres_to_genres[genre]]
            excluded = excluded_genres + excluded_subgenres
            # Construct the query string to include one genre and exclude some genres
            included_genre_query = "genre:\"{}\"".format(genre)
            
            # If you want to add exclude criteria in query
            #excluded_genres_query = "".join([" NOT genre:\"{}\"".format(gen) for gen in excluded_genres])
            #genre_query = "{}{}".format(included_genre_query, excluded_genres_query)
            
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
            time.sleep(2)
            songs_info = json.loads(song_request.text)['tracks']['items']
            
            for song_info in songs_info:
                id = song_info['id']
                if id in TracksIds:
                    continue
                else:
                    genres_track = list()
                    # Search for album genre
                    album_id = song_info['album']['id']
                    genre_request = requests.get(
                        '{}/albums/{}'.format(
                            SPOTIFY_API_URL,
                            album_id
                        ),
                        headers=authorization_header
                    )

                    time.sleep(2)
                    album_genre = json.loads(genre_request.text)['genres']
                    genres_track.extend(album_genre)
                    # Search for artists genre
                    for artist in song_info['artists']:
                        artist_id = artist['id']
                        genre_request = requests.get(
                            '{}/artists?ids={}'.format(
                                SPOTIFY_API_URL,
                                artist_id
                            ),
                            headers=authorization_header
                        )
                        
                        time.sleep(2)
                        artist_genre = json.loads(genre_request.text)['artists'][0]['genres']  
                        genres_track.extend(artist_genre)
                    # Check if an excluded genre is in the genres of the track 
                    break_outer_loop_2 = False    
                    for genre_track in genres_track:
                        for exculded_genre in excluded:
                            if(exculded_genre in genre_track):
                                break_outer_loop_2 = True
                                break
                        if break_outer_loop_2:
                            break
                    if break_outer_loop_2:
                        continue
                    
                    artist = song_info['artists'][0]['name']
                    artist_id = song_info['artists'][0]['id']
                    album = song_info['album']['name']
                    album_id = song_info['album']['id']
                    song = song_info['name']
                    popularity = song_info['popularity'] 
                    release_date = song_info['album']['release_date'] 
                    duration = song_info['duration_ms']/1000
                    preview_url = song_info['preview_url']
                    break
        except IndexError as e:
            print(e)
        except KeyError as e:
            print(e)
        except json.JSONDecodeError:
            print(genre_request.text)
            time.sleep(2)
        
    if song is None:
        artist = None
        song = None
        id = None
        preview_url = None
        artist_id = None
        album = None
        album_id = None
        release_date = None
        duration = None
        popularity = None
              
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

        TracksIds = []
        TracksIndexes = []
        TracksArtist = []
        TracksArtistId = []
        TracksPopularity = []
        TracksAlbum = []
        TracksAlbumId = []
        TracksReleaseDate = []
        TracksDuration = []
        TracksName = []
        TracksGenre = []
        TracksPreview = []  
        p = 0

        # Get a Spotify API token
        access_token = get_token()  
        consecutive_none_count = 0
        offset = 0
        count = 0
        number_of_genres = len(subgenres)
        consecutive_none_count = 0

        while(count < 10):
            if(count%20==0):
                offset = offset + 20
            for selected_genre in subgenres:
                song,artist,artist_id,id,preview_url,album,album_id,release_date,duration,popularity = request_valid_song(access_token, genre=selected_genre,offset=offset)
                if id is not None and preview_url is not None and id not in TracksIds:
                    TracksIds.append(id)
                    TracksName.append(song)
                    TracksArtist.append(artist)
                    TracksGenre.append(selected_genre)
                    TracksPreview.append(preview_url)
                    TracksArtistId.append(artist_id)
                    TracksAlbum.append(album)
                    TracksAlbumId.append(album_id)
                    TracksReleaseDate.append(release_date)
                    TracksDuration.append(duration)
                    TracksPopularity.append(popularity)
                    TracksIndexes.append(p)  
                    p += 1
                    print("{} Found a song: {} by {} with id: {} for genre: {}".format(p,song, artist, id, selected_genre))
                else:
                    print("Did not find a song for genre: {}".format(selected_genre))
                    subgenres.remove(selected_genre)  # Remove genre from list
                    consecutive_none_count += 1  # Increase the count of consecutive None results
            if(consecutive_none_count == number_of_genres):
                break
            count = count + 1
        filename = "results.txt"

        if not os.path.exists(filename):
            with open(filename, 'w') as file:
                file.write("This is a new file created!\n")
        with open(f'{filename}/tracks-ids.pkl', 'wb') as output_file:
            pickle.dump(TracksIds, output_file)
        with open(f'{filename}/tracks-indexes.pkl', 'wb') as output_file:
            pickle.dump(TracksIndexes, output_file)

        zippedList =  list(zip(TracksIndexes, TracksName, TracksIds, TracksArtist, TracksArtistId, TracksGenre, TracksPreview, TracksAlbum, TracksAlbumId, TracksReleaseDate, TracksDuration, TracksPopularity))

        df = pd.DataFrame(zippedList, columns = ['Index', 'Name' , 'ID', 'Artist', "Artist's ID", 'Genre',  'Preview', 'Album', "Album's ID", "Release Date", "Duration", "Popularity"], index=TracksIndexes) 
        
        df.to_csv(f'{filename}/tracks.csv',index=False)
        send_email("Script Completed ", "Your Python script has finished successfully.")
    except Exception as e:
        send_email("Script Error", f"An error occurred in your Python script:\n\n{str(e)}")