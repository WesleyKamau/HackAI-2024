

def styleSwap(content_image_url, style_image_url):
    import tensorflow as tf
    import numpy as np
    from PIL import Image
    from tensorflow import keras
    from tensorflow.keras import layers


    # Path to the directory containing the SavedModel
    saved_model_path = '/home/wesleykamau/helloWorld/model/'

    # Load the model
    model = tf.saved_model.load(saved_model_path)

    # Use the model for inference or other tasks


    # Load and preprocess the content and style images
    content_image = load_and_preprocess_image(content_image_url)
    style_image = load_and_preprocess_image(style_image_url)

    # Perform style transfer
    stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]

    # Convert the tensor to an image
    stylized_image = tensor_to_image(stylized_image)

    return stylized_image

def load_and_preprocess_image(image_url):
    # Load and preprocess the image
    img = tf.keras.utils.load_img(image_url, target_size=(224, 224))
    img = tf.keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

from PIL import Image
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"



client_id = "702428c1f47241b7b19f4c718ac63cdf"
client_secret = "823173ff77c5420783ff900f17ee23ce"

# Initialize Spotipy with your credentials
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Search for the artist
artist = input("Enter the name of the artist: ")
results = sp.search(q='artist:' + artist, type='artist')
items = results['artists']['items']

artist = items[0]
artist_id = artist['id']
    
# Get the artist's albums
albums = sp.artist_albums(artist_id, album_type='album')
    
# Extract album covers
album_covers = [album['images'][0]['url'] for album in albums['items']]
    
scale = 1500//len(album_covers)

img_dimension = scale*len(album_covers)
new = Image.new("RGBA", (img_dimension,img_dimension))

## Rows are the source
## Columns are the style

for x in range(len(album_covers)):
    for y in range(len(album_covers)):
        if x == y:
            r = requests.get(album_covers[x], stream=True)
            print(album_covers[x])
            img = Image.open(r.raw)
        else:
            img = styleSwap(album_covers[x], album_covers[y])
        img = img.resize((scale,scale))
        new.paste(img, (scale*x,scale*y))
            


new.save("current-grid.png")

# new = Image.new("RGBA", (1000,1000))
# img = Image.open("images.jpg")
# img = img.resize((500,500))
# new.paste(img, (0,0))
# new.paste(img, (500,500))

# new.save("new.png")