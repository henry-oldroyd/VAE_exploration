import tensorflow as tf


def VAE_loss(model, y_true, y_pred):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)

def create_VAE_model(encoder_model, decoder_model):


    VAE_model_input = tf.keras.Input(input_shape=(28, 28, 1))

    VAE_model_output = encoder_model(VAE_model)
    VAE_model_output = decoder_model(VAE_model)

    VAE_model = tf.keras.Model(
        inputs = VAE_model_input,
        outputs = VAE_model_output
    )



