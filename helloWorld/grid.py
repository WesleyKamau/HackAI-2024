
def styleSwap(content_image_url,style_image_url,scale):
    import os
    import tensorflow as tf
    tf.keras.backend.clear_session()
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    # Load compressed models from tensorflow_hub
    os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

    import IPython.display as display

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams['figure.figsize'] = (12, 12)
    mpl.rcParams['axes.grid'] = False

    import numpy as np
    import PIL.Image
    import time
    import functools

    def tensor_to_image(tensor):
        tensor = tensor*255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor)>3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return PIL.Image.fromarray(tensor)

    content_path = tf.keras.utils.get_file(origin=content_image_url)
    style_path = tf.keras.utils.get_file(origin=style_image_url)

    def load_img(path_to_img,size):
        max_dim = size
        img = tf.io.read_file(path_to_img)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim

        print(shape*scale)
        new_shape = tf.cast(shape * scale, tf.int32)

        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img

    def imshow(image, title=None):
        if len(image.shape) > 3:
            image = tf.squeeze(image, axis=0)

        plt.imshow(image)
        if title:
            plt.title(title)

    content_image = load_img(content_path,scale)
    style_image = load_img(style_path,scale)


    plt.subplot(1, 2, 1)
    imshow(content_image, 'Content Image')

    plt.subplot(1, 2, 2)
    imshow(style_image, 'Style Image')

    plt.savefig("initial.png")

    x = tf.keras.applications.vgg19.preprocess_input(content_image*255)
    x = tf.image.resize(x, (224, 224))
    vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
    prediction_probabilities = vgg(x)
    prediction_probabilities.shape


    predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(prediction_probabilities.numpy())[0]
    [(class_name, prob) for (number, class_name, prob) in predicted_top_5]


    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

    print()
    for layer in vgg.layers:
        print(layer.name)


    content_layers = ['block5_conv2'] 

    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1', 
                    'block4_conv1', 
                    'block5_conv1']

    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)

    def vgg_layers(layer_names):
        """ Creates a VGG model that returns a list of intermediate output values."""
        # Load our model. Load pretrained VGG, trained on ImageNet data
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        
        outputs = [vgg.get_layer(name).output for name in layer_names]

        model = tf.keras.Model([vgg.input], outputs)
        return model

    style_extractor = vgg_layers(style_layers)
    style_outputs = style_extractor(style_image*255)

    #Look at the statistics of each layer's output
    for name, output in zip(style_layers, style_outputs):
        print(name)
        print("  shape: ", output.numpy().shape)
        print("  min: ", output.numpy().min())
        print("  max: ", output.numpy().max())
        print("  mean: ", output.numpy().mean())
        print()

    def gram_matrix(input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
        return result/(num_locations)

    class StyleContentModel(tf.keras.models.Model):
        def __init__(self, style_layers, content_layers):
            super(StyleContentModel, self).__init__()
            self.vgg = vgg_layers(style_layers + content_layers)
            self.style_layers = style_layers
            self.content_layers = content_layers
            self.num_style_layers = len(style_layers)
            self.vgg.trainable = False

        def call(self, inputs):
            "Expects float input in [0,1]"
            inputs = inputs*255.0
            preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
            outputs = self.vgg(preprocessed_input)
            style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                            outputs[self.num_style_layers:])

            style_outputs = [gram_matrix(style_output)
                            for style_output in style_outputs]

            content_dict = {content_name: value
                            for content_name, value
                            in zip(self.content_layers, content_outputs)}

            style_dict = {style_name: value
                        for style_name, value
                        in zip(self.style_layers, style_outputs)}

            return {'content': content_dict, 'style': style_dict}
        

    extractor = StyleContentModel(style_layers, content_layers)

    results = extractor(tf.constant(content_image))

    print('Styles:')
    for name, output in sorted(results['style'].items()):
        print("  ", name)
        print("    shape: ", output.numpy().shape)
        print("    min: ", output.numpy().min())
        print("    max: ", output.numpy().max())
        print("    mean: ", output.numpy().mean())
        print()

    print("Contents:")
    for name, output in sorted(results['content'].items()):
        print("  ", name)
        print("    shape: ", output.numpy().shape)
        print("    min: ", output.numpy().min())
        print("    max: ", output.numpy().max())
        print("    mean: ", output.numpy().mean())

    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    image = tf.Variable(content_image)

    def clip_0_1(image):
        return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    style_weight=1e-1
    content_weight=1e4

    def style_content_loss(outputs):
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                            for name in style_outputs.keys()])
        style_loss *= style_weight / num_style_layers

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                                for name in content_outputs.keys()])
        content_loss *= content_weight / num_content_layers
        loss = style_loss + content_loss
        return loss

    @tf.function()
    def train_step(image):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = style_content_loss(outputs)

        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))

    train_step(image)
    train_step(image)
    train_step(image)
    tensor_to_image(image)

    import time
    start = time.time()

    epochs = 10
    steps_per_epoch = 100

    step = 0
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            train_step(image)
            print(".", end='', flush=True)
        display.clear_output(wait=True)
        display.display(tensor_to_image(image))
        print("Train step: {}".format(step))
    
    end = time.time()
    print("Total time: {:.1f}".format(end-start))

    def high_pass_x_y(image):
        x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
        y_var = image[:, 1:, :, :] - image[:, :-1, :, :]

        return x_var, y_var
        

    x_deltas, y_deltas = high_pass_x_y(content_image)

    plt.figure(figsize=(14, 10))
    plt.subplot(2, 2, 1)
    imshow(clip_0_1(2*y_deltas+0.5), "Horizontal Deltas: Original")

    plt.subplot(2, 2, 2)
    imshow(clip_0_1(2*x_deltas+0.5), "Vertical Deltas: Original")

    x_deltas, y_deltas = high_pass_x_y(image)

    plt.subplot(2, 2, 3)
    imshow(clip_0_1(2*y_deltas+0.5), "Horizontal Deltas: Styled")

    plt.subplot(2, 2, 4)
    imshow(clip_0_1(2*x_deltas+0.5), "Vertical Deltas: Styled")

    plt.figure(figsize=(14, 10))

    sobel = tf.image.sobel_edges(content_image)
    plt.subplot(1, 2, 1)
    imshow(clip_0_1(sobel[..., 0]/4+0.5), "Horizontal Sobel-edges")
    plt.subplot(1, 2, 2)
    imshow(clip_0_1(sobel[..., 1]/4+0.5), "Vertical Sobel-edges")


    def total_variation_loss(image):
        x_deltas, y_deltas = high_pass_x_y(image)
        return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))
        

    total_variation_loss(image).numpy()

    tf.image.total_variation(image).numpy()

    total_variation_weight=60

    @tf.function()
    def train_step(image):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = style_content_loss(outputs)
            loss += total_variation_weight*tf.image.total_variation(image)

        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))

    opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
    image = tf.Variable(content_image)
        
    import time
    start = time.time()

    epochs = 10
    steps_per_epoch = 100

    step = 0
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            train_step(image)
            print(".", end='', flush=True)
        display.clear_output(wait=True)
        display.display(tensor_to_image(image))
        print("Train step: {}".format(step))

    end = time.time()
    print("Total time: {:.1f}".format(end-start))

    mpl.pyplot.close()
    tf.keras.backend.clear_session()

    # file_name = 'stylized-image.png'
    # tensor_to_image(image).save(file_name)

    # try:
    #     from google.colab import files
    # except ImportError:
    #     pass
    # else:
    #     files.download(file_name)

    return tensor_to_image(image)


from PIL import Image
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests
import os



client_id = "702428c1f47241b7b19f4c718ac63cdf"
client_secret = "823173ff77c5420783ff900f17ee23ce"

# Initialize Spotipy with your credentials
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Search for the artist
# artist = input("Enter the name of the artist: ")
artist = "Taylor Swift"
results = sp.search(q='artist:' + artist, type='artist')
items = results['artists']['items']

artist = items[0]
artist_id = artist['id']
    
# Get the artist's albums
albums = sp.artist_albums(artist_id, album_type='album')
print(albums)
    
# Assuming you have the album_covers array and the albums data

# Filter out albums that do not have "deluxe" in their titles
filtered_albums = [album for album in albums['items'] if 'deluxe' not in album['name'].lower()]

# Now, extract the album covers for the filtered albums
album_covers = [album['images'][0]['url'] for album in filtered_albums]

xmin = 0
ymin = 0
resume = "Y"

if os.path.exists('xy_values.txt'):
    while (resume != "Y" and resume != "N"):
        print("lol")
        # resume = input("Would you like to resume from the last saved image? (Y/N): ")

    if resume == "Y":
        with open('xy_values.txt', 'r') as file:
            line = file.readline()
            parts = line.split(',')

            # Extract x and y values
            xmin = parts[0].split(': ')[1]
            ymin = parts[1].split(': ')[1]

            print(f'xmin: {xmin}, ymin: {ymin}')

scale = 3000//len(album_covers)

img_dimension = scale*len(album_covers)
if xmin == 0 and ymin == 0:
    new = Image.new("RGBA", (img_dimension,img_dimension))

    ## Rows are the source
    ## Columns are the style
    new.save("current-grid.png")
    del new

founditem = False
for x in range(len(album_covers)):
    for y in range(len(album_covers)):
        if x>=int(xmin) and y>=int(ymin):
            founditem = True
            if x == y:
                r = requests.get(album_covers[x], stream=True)
                print(album_covers[x])
                img = Image.open(r.raw)
                del r
            else:
                img = styleSwap(album_covers[x], album_covers[y],scale)
            img = img.resize((scale,scale))
            new=Image.open("current-grid.png")
            new.paste(img, (scale*x,scale*y))
            new.save("current-grid.png")
            # Open the file in write mode, which clears it
            with open('xy_values.txt', 'w'):
                pass
            with open('xy_values.txt', 'a') as file:
                file.write(f'x: {x}, y: {y}')
            del new
            del img
    if founditem:
        ymin = 0


# Specify the file path
file_path = 'xy_values.txt'

# Check if the file exists and delete it
if os.path.exists(file_path):
    os.remove(file_path)
else:
    print(f"The file {file_path} does not exist")  
            

# new = Image.new("RGBA", (1000,1000))
# img = Image.open("images.jpg")
# img = img.resize((500,500))
# new.paste(img, (0,0))
# new.paste(img, (500,500))

# new.save("new.png")