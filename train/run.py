import train.config as con
import model.model_fn as model_fn
import data.data_fn as data_input
import numpy as np
import tensorflow as tf
import tensorflow.estimator as tfe
tf.logging.set_verbosity(tf.logging.INFO)
from . import config as _con
args=_con.parse_args()


def norm_progress():
    model_params = {"learning_rate": args.learning_rate,'model_dir':args.model_s_dir}
    config=tfe.RunConfig(model_dir=args.model_s_dir,save_summary_steps=10)
    network = tf.estimator.Estimator(
        model_fn=model_fn.patch_segmentation_fn,
        model_dir=args.model_s_dir,
        config=config,
        params=model_params)
    input_fn,data,label= data_input.MA_segmention_input()
    print('num of data:',len(label))
    print('batch_size:',args.batch_size)
    print('n_epoch:',args.n_epoch)
    print(args)
    for epoch in range(args.n_epoch):
        input_fn,data,label= data_input.MA_segmention_input()
        print('=========={}/{}=========='.format(epoch,args.n_epoch))
        network.train(input_fn=input_fn,steps=len(label)//args.batch_size)



def my_gan_progress():

    g_model_params = {"learning_rate": args.learning_rate,'model_dir':args.model_g_dir}
    d_model_params = {"learning_rate": args.learning_rate,'model_dir':args.model_d_dir}

    g_config=tfe.RunConfig(model_dir=args.model_g_dir,save_summary_steps=10)
    d_config=tfe.RunConfig(model_dir=args.model_d_dir,save_summary_steps=10)

    g_network = tf.estimator.Estimator(
        model_fn=model_fn.patch_generator_fn,
        model_dir=args.model_g_dir,
        config=g_config,
        params=g_model_params)

    d_network = tf.estimator.Estimator(
        model_fn=model_fn.patch_discriminator_fn,
        model_dir=args.model_d_dir,
        config=d_config,
        params=d_model_params)

    g_input_fn,data,label = data_input.train_input_fn_layer_segmention()

    print('num of data:',len(label))
    print('batch_size:',args.batch_size)
    print('n_epoch:',args.n_epoch)
    print(args)
    for epoch in range(args.n_epoch):
        train_input_fn,data,label = data_input.train_input_fn_layer_segmention()
        print('=========={}/{}=========='.format(epoch,args.n_epoch))
        nn.train(input_fn=train_input_fn,steps=len(label)//args.batch_size)


def es_gan_progress():
    tfgan = tf.contrib.gan
    g_model_params = {"learning_rate": args.learning_rate,'model_dir':args.model_g_dir}
    d_model_params = {"learning_rate": args.learning_rate,'model_dir':args.model_d_dir}

    g_config=tfe.RunConfig(model_dir=args.model_g_dir,save_summary_steps=10)
    d_config=tfe.RunConfig(model_dir=args.model_d_dir,save_summary_steps=10)


    gan_model = tfgan.gan_model(
        generator_fn=model_fn.g_patch,
        discriminator_fn=model_fn.d_patch,
        real_data=real_images,
        generator_inputs=distorted_images)

    tfgan.train.add_image_comparison_summaries(gan_model,num_comparisons=3, display_diffs=True)
    tfgan.train.add_gan_model_image_summaries(gan_model, grid_size=3)
    
    with tf.name_scope('losses'):
        gan_loss = tfgan.gan_loss(
            gan_model,
            generator_loss_fn=tfgan.losses.least_squares_generator_loss,
            discriminator_loss_fn=tfgan.losses.least_squares_discriminator_loss)
        l1_pixel_loss = tf.norm(gan_model.real_data - gan_model.generated_data,ord=1) / FLAGS.patch_size ** 2
        gan_loss = tfgan.losses.combine_adversarial_loss(gan_loss, gan_model, l1_pixel_loss,weight_factor=FLAGS.weight_factor)


