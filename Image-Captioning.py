import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM

# Load pre-trained InceptionV3 model
image_model = InceptionV3(weights='imagenet')
image_model = Model(image_model.input, image_model.layers[-2].output)

# Load pre-trained captioning model
caption_model = tf.keras.models.load_model('caption_model.h5')

# Load word-to-index and index-to-word mappings
word_to_index = np.load('word_to_index.npy', allow_pickle=True).item()
index_to_word = np.load('index_to_word.npy', allow_pickle=True).item()

# Load tokenizer used for captions
tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(
    open('tokenizer.json').read())

# Maximum sequence length for captions
max_sequence_length = 30


def preprocess_image(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(299, 299))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def encode_image(image):
    # Encode the image using InceptionV3 model
    image = preprocess_image(image)
    features = image_model.predict(image)
    features = tf.reshape(features, shape=(1, -1))
    return features


def generate_caption(image):
    # Encode the image
    image = encode_image(image)

    # Initialize the caption with the start token
    caption_input = tokenizer.texts_to_sequences(['<start>'])[0]
    caption = [caption_input]

    # Generate the caption word by word
    for _ in range(max_sequence_length):
        caption_input = pad_sequences([caption_input], maxlen=max_sequence_length, padding='post')

        # Predict the next word
        predictions = caption_model.predict([image, caption_input])
        predicted_index = np.argmax(predictions[0, -1, :])
        predicted_word = index_to_word[predicted_index]

        # Break if the end token is predicted
        if predicted_word == '<end>':
            break

        # Append the predicted word to the caption
        caption[0].append(predicted_index)
        caption_input = caption[0]

    # Convert the predicted caption from indices to words
    predicted_caption = [index_to_word[word_index] for word_index in caption[0]]
    predicted_caption = ' '.join(predicted_caption[1:-1])  # Exclude start and end tokens
    return predicted_caption


# Example usage
image_path = 'example.jpg'
caption = generate_caption(image_path)
print(caption)
