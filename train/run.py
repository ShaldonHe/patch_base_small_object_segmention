import train.config as con
import model.model_fn as model_fn
import data.data_fn as data_input
import numpy as np
import tensorflow as tf
import tensorflow.estimator as tfe
tf.logging.set_verbosity(tf.logging.INFO)

def gan_progress():
    tfgan = tf.contrib.gan
    gan_c=con.parse_args()
    # Create GAN estimator.
    gan_estimator = tfgan.estimator.GANEstimator(
        gan_c.model_dir,
        generator_fn=model_fn.g_patch,
        discriminator_fn=model_fn.d_patch,
        generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
        discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
        generator_optimizer=tf.train.AdamOptimizer(0.1, 0.5),
        discriminator_optimizer=tf.train.AdamOptimizer(0.1, 0.5))


    train_input=data_input.train_input_fn_layer_segmention()
    # Train estimator.
    gan_estimator.train(train_input, gan_c.n_epoch)

    # Evaluate resulting estimator.
#     gan_estimator.evaluate(eval_input_fn)

#     # Generate samples from generator.
#     predictions = np.array([
#         x for x in gan_estimator.predict(predict_input_fn)])
