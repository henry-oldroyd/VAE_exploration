# %%
# import modules
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, LeakyReLU, Input, BatchNormalization, Softmax, Reshape, Conv2DTranspose, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.initializers import RandomNormal
import matplotlib.pyplot as plt

# %%
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# %%
train_images[0].shape

# %%
num_train_real_data_items = train_images.shape[0]
num_test_real_data_items = test_images.shape[0]

# %%
# activation_funcion = Activation("tanh")
# activation_funcion = Activation("sigmoid")
activation_funcion = LeakyReLU(alpha=0.2)
# use_normalisation = True
use_normalisation = False
use_dropout = False
dropout_rate = 0.2

# %%
input_shape = (28, 28, 1)
input_vector_size = input_shape[0] * input_shape[1]
latent_space_size = 50

# hidden_neurons_per_hidden_layer = [200, 100]
hidden_neurons_per_hidden_layer = [100]

encoder_neurons_per_hidden_layer = hidden_neurons_per_hidden_layer
decoder_neurons_per_hidden_layer = list(reversed(hidden_neurons_per_hidden_layer))

# %%
model_Encoder = Sequential()

model_Encoder.add(Input(input_shape))
model_Encoder.add(Flatten())

if use_normalisation:
    model_Encoder.add(BatchNormalization())

for hidden_layer_num_neurons in encoder_neurons_per_hidden_layer:
    model_Encoder.add(Dense(hidden_layer_num_neurons))

    model_Encoder.add(activation_funcion)

    if use_normalisation:
        model_Encoder.add(BatchNormalization())

    if use_dropout:
        model_Encoder.add(Dropout(dropout_rate))


model_Encoder.add(Dense(latent_space_size))
model_Encoder.add(Activation("tanh"))


# %%
model_Encoder.summary()

# %%
model_Decoder = Sequential()

model_Decoder.add(Input(latent_space_size))

if use_normalisation:
    model_Encoder.add(BatchNormalization())

for hidden_layer_num_neurons in decoder_neurons_per_hidden_layer:
    model_Decoder.add(Dense(hidden_layer_num_neurons))

    model_Decoder.add(activation_funcion)

    if use_normalisation:
        model_Decoder.add(BatchNormalization())

    if use_dropout:
        model_Decoder.add(Dropout(dropout_rate))

model_Decoder.add(Dense(input_vector_size))
model_Decoder.add(Activation("sigmoid"))
model_Decoder.add(Reshape(input_shape))

# %%
model_Decoder.summary()

# %%
autoencoder_input = Input(input_shape)
encoded_output = model_Encoder(autoencoder_input)
decoded_output = model_Decoder(encoded_output)

model_autoencoder = Model(autoencoder_input, decoded_output)

# %%
model_autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# model_autoencoder.compile(optimizer='adam', loss='MSE')

# %%
model_autoencoder.fit(
    x=train_images,
    y=train_images,
    batch_size = 50,
    epochs = 5,
    verbose = 2,
    validation_split = 0.2
)

# %%



