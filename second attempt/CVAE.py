import tensorflow as tf

class Conditional_Encoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim

        self.input_shape_image = (None, 28, 28, 1)
        self.input_shape_label = (None, 10)


        self.convolutional_component_model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten()
        ])

        self.dense_component_model = tf.keras.Sequential([
            tf.keras.layers.Dense(2 * self.latent_dim)
        ])

    def build(self):
        super().build((self.input_shape_image, self.input_shape_label))

        self.convolutional_component_model.build(self.input_shape_image)
        self.dense_component_model.build(
            (None, self.convolutional_component_model.output_shape[-1] + self.input_shape_label[-1])
        )
    
    # def compute_output_shape(self, input_shape):
    #     # Return the output shape as a tuple
    #     return ((input_shape[0], self.latent_dim), (input_shape[0], self.latent_dim))


    def summary(self):
        super().summary()
        self.convolutional_component_model.summary()
        self.dense_component_model.summary()

    def call(self, inputs):
        # separate out the image and label
        image, label = inputs

        image = tf.reshape(image, (-1, 28, 28, 1))

        # process the image to extract features
        convolutional_features = self.convolutional_component_model(image)

        # concatenate the label to the features
        combined_features = tf.concat([convolutional_features, label], axis=-1)

        # pass through dense layer
        encoder_output = self.dense_component_model(combined_features)

        # split the output into mean and ln variance
        mean, ln_var = tf.split(encoder_output, num_or_size_splits=2, axis=-1)
        variance = tf.exp(ln_var)

        return mean, variance

def dimension_product(iterable):
    result = 1
    for item in iterable:
        if item:
            result *= item
    return result


class Conditional_Decoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim

        self.latent_input_shape = (None, self.latent_dim)
        self.label_input_shape = (None, 10)
        self.combined_input_shape = (None, self.latent_input_shape[-1] + self.label_input_shape[-1])

        self.convolutional_transpose_component_model_input_shape = (None, 7, 7, 32)
        self.dense_component_model_output_shape = (None, dimension_product(self.convolutional_transpose_component_model_input_shape))


        self.dense_component_model = tf.keras.Sequential([
            tf.keras.layers.Dense(self.dense_component_model_output_shape[-1], activation='relu'),
        ])

        self.convolutional_transpose_component_model = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding='same',
                activation='relu'
            ),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=2, padding='same',
                activation='relu'
            ),
            # No activation
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=1, padding='same'
            )
        ])



    def build(self):
        super().build((self.latent_input_shape, self.label_input_shape))

        self.dense_component_model.build(self.combined_input_shape)

        assert self.dense_component_model.output_shape  == self.dense_component_model_output_shape
        self.convolutional_transpose_component_model.build(self.convolutional_transpose_component_model_input_shape)

    # def compute_output_shape(self, input_shape):
    #     # Return the output shape as a tuple
    #     output_shape = list(self.convolutional_transpose_component_model_output_shape[-1])
    #     output_shape[0] = input_shape[0]
    #     return tuple(output_shape)

    def summary(self):
        super().summary()
        self.dense_component_model.summary()
        self.convolutional_transpose_component_model.summary()

    def call(self, inputs):
        # separate out the image and label
        latent_vector, label = inputs
        combined_inputs = tf.concat([latent_vector, label], axis=-1)

        # pass through dense layer
        dense_output = self.dense_component_model(combined_inputs)

        # reshape from flat vector to matrix

        convolutional_transpose_input = tf.reshape(
            dense_output, 
            tuple(dimension if dimension else -1 for dimension in self.convolutional_transpose_component_model_input_shape)
        )

        # pass through convolutional transpose layers
        output_image_logits = self.convolutional_transpose_component_model(convolutional_transpose_input)

        return output_image_logits
    


class Conditional_VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = Conditional_Encoder(latent_dim)
        self.decoder = Conditional_Decoder(latent_dim)

    def summary(self):
        super().summary()
        # self.encoder.summary()
        # self.decoder.summary()


    def build(self):
        super().build((self.encoder.input_shape_image, self.encoder.input_shape_label))
        self.encoder.build()
        self.decoder.build()


    def call(self, inputs):
        # image, label = inputs
        image = inputs[0]
        label = inputs[1]

        mean, variance = self.encoder((image, label))

        # sample from the distribution
        epsilon = tf.random.normal(tf.shape(mean))
        latent_vector = mean + tf.math.sqrt(variance) * epsilon

        output_image_logits = self.decoder((latent_vector, label))
        output_image = tf.sigmoid(output_image_logits)

        return output_image
    
    def fit(self, x, y, batch_size, epochs):
        super().fit(x, y, batch_size, epochs)



