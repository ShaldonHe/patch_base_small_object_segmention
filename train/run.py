import train.config as con
import model.model_fn as model_fn
import data.data_fn as data_input
import numpy as np
import tensorflow as tf
import tensorflow.estimator as tfe
tf.logging.set_verbosity(tf.logging.INFO)

def progress():
    args=con.parse_args()
    model_params = {"learning_rate": args.learning_rate,'model_dir':args.model_dir}
    run_config=tfe.RunConfig(
            model_dir=args.model_dir,
            save_summary_steps=10)
    nn = tf.estimator.Estimator(
            model_fn=model_fn.Layer_Segmention_fn,
            model_dir=args.model_dir,
            config=run_config,
            params=model_params)
    for epoch in range(args.n_epoch):
        train_input_fn,data,label = data_input.train_input_fn_segmention()
        print('num of data:',len(label))
        print('batch_size:',args.batch_size)

        # test_input_fn,data = data_input.test_input_fn_segmention()
        nn.train(input_fn=train_input_fn,steps=np.ceil(len(label)/args.batch_size))


def activate_progress():
    args=con.parse_args()
    model_params = {"learning_rate": args.learning_rate,'model_dir':args.model_dir}
    run_config=tfe.RunConfig(
            model_dir=args.model_dir,
            save_summary_steps=10)
    nn = tf.estimator.Estimator(
            model_fn=model_fn.Layer_Segmention_fn,
            model_dir=args.model_dir,
            config=run_config,
            params=model_params)
    for epoch in range(args.n_epoch):
        train_input_fn,data,label = data_input.train_input_fn_segmention()
        nn.train(input_fn=train_input_fn,steps=len(label)//args.batch_size)
        r_predict=nn.predict(input_fn=train_input_fn)
        for num,r in enumerate(r_predict) :
            n_label=label[num][0]
            r_predict=r['predict']
            shape_size=n_label.shape[0]
            center_label=n_label[shape_size//4:(shape_size-shape_size//4),shape_size//4:(shape_size-shape_size//4),:]
            center_predict=r_predict[shape_size//4:shape_size-shape_size//4,shape_size//4:shape_size-shape_size//4,:]
            print(np.mean(np.abs(center_label-center_predict)))
            value[num]=np.mean(np.abs(center_label-center_predict))
            if num >300:
                break
        np.savez_compressed('./data/model_data/e_loss.npz',ids=label._index2ids,e_loss=value)


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
