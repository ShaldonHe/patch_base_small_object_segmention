import train.config as con
import model.model_fn as model_fn
import data.data_fn as data_input
import numpy as np
import tensorflow as tf
import tensorflow.estimator as tfe
tf.logging.set_verbosity(tf.logging.INFO)
from . import config as _con
args=_con.parse_args()
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def norm_progress():
    model_params = {"learning_rate": args.learning_rate,'model_dir':args.model_s_dir}
    session_config = tf.ConfigProto(log_device_placement=True)
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config=tfe.RunConfig(model_dir=args.model_s_dir,save_summary_steps=10)
    config = config.replace(session_config=session_config)
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

def activate_progress():
    import libs.common.data_interface as libdi
    import libs.model.losses as libms
    model_params = {"learning_rate": args.learning_rate,'model_dir':args.model_s_dir}
    # session_config = tf.ConfigProto(log_device_placement=True)
    # session_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config=tfe.RunConfig(model_dir=args.model_s_dir,save_summary_steps=10)
    # config = config.replace(session_config=session_config)
    network = tf.estimator.Estimator(model_fn=model_fn.patch_segmentation_fn,model_dir=args.model_s_dir,config=config,params=model_params)
    print(args)
    p=0.5

    data,label= data_input.MA_segmention_data()
    # data,label= data_input.MA_segmention_debug_data()
    rank_list=libdi.rank_list(len(label))
    for epoch in range(args.n_epoch):
        print('num of data:',len(label))
        print('batch_size:',args.train_batch_size)
        print('n_epoch:',args.n_epoch)
        new_index=rank_list.get_top(p)
        data.update_index(new_index)
        label.update_index(new_index)
        input_fn=tfe.inputs.numpy_input_fn(x={"images":data},y=label,batch_size=args.train_batch_size,shuffle=False)
        print('==== train:{}/{}=========='.format(epoch,args.n_epoch))
        network.train(input_fn=input_fn,steps=len(label)//args.train_batch_size)
        print('===== eval:{}/{}=========='.format(epoch,args.n_epoch))
        data.update_index()
        label.update_index()

        input_fn=tfe.inputs.numpy_input_fn(x={"images":data},y=label,batch_size=args.eval_batch_size,shuffle=False)
        r=network.predict(input_fn=input_fn)
        total_loss=0
        for i,_r in enumerate(r):
            l_loss=libms.np_iou_loss(_r['predict'][:,:,0],label[i][0][:,:,0])
            total_loss+=l_loss
            rank_list.update(i,l_loss)
        rank_list.sort()
        print('eval_loss:{}'.format(total_loss/len(label)))
        print('==========={}/{}=end======'.format(epoch,args.n_epoch))



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


