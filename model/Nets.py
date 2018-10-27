import tensorflow as tf
from tensorflow import estimator as tfe

import libs.model.blocks as _bk
import libs.model.layers as _l
import libs.model.ops as _op
from . import config as c
_c_g=c.parse_g_args()
_c_d=c.parse_d_args()
_c_s=c.parse_args()

def patch_segmentation(x, is_training):
    # input data shape
    # 256*256*(3:origin image )
    print('-'*50,'Input Node Name','-'*50)
    print(x.name)
    print('-'*120)
    k_size=3
    init_filter=32
    # model_parms=dict(kernel_size=(k_size,k_size),conv=_l.Conv.conv2d,norm=_l.Norm.batch,activation=_op.Activation.leak_relu)
    model_parms=dict(kernel_size=(k_size,k_size),conv=_l.Conv.conv2d,norm=_l.Norm.batch,activation=None)
    block_0=_bk.conv_norm_activation(x,filters=init_filter,**model_parms)
    block_1=_bk.res_v1_block(block_0,**model_parms) 
    block_2=_bk.res_v1_block(block_1,**model_parms) 
    block_3=_bk.res_v1_block(block_2,**model_parms) 
    block_4=_bk.res_v1_block(block_3,**model_parms) 
    unet_conv_t_4=_bk.ushape_block(block_4,block_3,img_size=block_3.shape[1:3].as_list(),filters=block_3.shape[3].value,**model_parms)
    unet_conv_t_3=_bk.ushape_block(unet_conv_t_4,block_2,img_size=block_2.shape[1:3].as_list(),filters=block_2.shape[3].value,**model_parms)
    unet_conv_t_2=_bk.ushape_block(unet_conv_t_3,block_1,img_size=block_1.shape[1:3].as_list(),filters=block_1.shape[3].value,**model_parms)
    unet_conv_t_1=_bk.ushape_block(unet_conv_t_2,block_0,img_size=block_0.shape[1:3].as_list(),filters=block_0.shape[3].value,**model_parms)
    unet_conv_t_0=_bk.ushape_block(unet_conv_t_1,x,img_size=x.shape[1:3].as_list(),filters=init_filter)
    result=_bk.conv_norm_activation(unet_conv_t_0,filters=_c_s.layer_num,kernel_size=(1,1),activation=None)
    # result=_bk.conv_norm_activation(unet_conv_t_0,filters=_c_s.layer_num,kernel_size=(1,1),activation=_op.Activation.relu)
    print('-'*50,'Result Node Name','-'*50)
    print(result.name)
    print('-'*120)
    return result



def patch_generator(x, is_training):
    # input data shape
    # 256*256*(3:origin image )
    print('-'*50,'Input Node Name','-'*50)
    print(x.name)
    print('-'*120)
    k_size=3
    p_size=2
    init_filter=16
    model_parms=dict(pool_size=[p_size,p_size],kernel_size=k_size,conv='separable_conv2d',norm='batch_norm',activation='relu')
    x=tf.reshape(x,[-1]+config_l.input_shape[:],name='Input.x')
    block_0=_bk.conv_norm_activation(x,filters=init_filter,**model_parms)
    block_1=_bk.res_v1_block(block_0,**model_parms) 
    block_2=_bk.res_v1_block(block_1,**model_parms) 
    block_3=_bk.res_v1_block(block_2,**model_parms) 
    block_4=_bk.res_v1_block(block_3,**model_parms) 
    unet_conv_t_4=_bk.unet_conv2d_T_block_v0(block_4,block_3,img_size=block_3.shape[1:3].as_list(),filters=block_3.shape[3].value,**model_parms)
    unet_conv_t_3=_bk.unet_conv2d_T_block_v0(unet_conv_t_4,block_2,img_size=block_2.shape[1:3].as_list(),filters=block_2.shape[3].value,**model_parms)
    unet_conv_t_2=_bk.unet_conv2d_T_block_v0(unet_conv_t_3,block_1,img_size=block_1.shape[1:3].as_list(),filters=block_1.shape[3].value,**model_parms)
    unet_conv_t_1=_bk.unet_conv2d_T_block_v0(unet_conv_t_2,block_0,img_size=block_0.shape[1:3].as_list(),filters=block_0.shape[3].value,**model_parms)
    unet_conv_t_0=_bk.unet_conv2d_T_block_v0(unet_conv_t_1,x,img_size=x.shape[1:3].as_list(),filters=init_filter)
    result=_bk.conv_norm_activation(unet_conv_t_0,filters=config_l.layer_num,kernel_size=(1,1),**model_parms)
    print('-'*50,'Result Node Name','-'*50)
    print(result.name)
    print('-'*120)
    return tf.concat([x,result],concat_dim=3)

def patch_discriminator(x,is_training):
    # input data shape
    # 256*256*(3:origin image 1:MA heatmap)
    print('-'*50,'Input Node Name','-'*50)
    print(x.name)
    print('-'*120)
    k_size=3
    p_size=2
    init_filter=24
    model_parms=dict(pool_size=[p_size,p_size],kernel_size=k_size,conv='separable_conv2d',norm='batch_norm',activation='relu')
    x=tf.reshape(x,[-1]+config_l.input_shape[:],name='Input.x')
    block_0=_bk.conv_norm_activation(x,filters=init_filter,**model_parms)
    # tf
    block_1=_bk.res_v1_block(block_0,**model_parms) 
    block_2=_bk.res_v1_block(block_1,**model_parms) 
    block_3=_bk.res_v1_block(block_2,**model_parms) 
    block_4=_bk.res_v1_block(block_3,**model_parms)
    block_5=_bk.res_v1_block(block_4,**model_parms)
    block_6=_bk.res_v1_block(block_5,**model_parms)

    _y=_l.conv(block_6,_l.Conv.conv2d,filters=2,kernel_size=(1,1))
    _y=_l.pooling(_y,pooling=_l.Pooling.global_avg_2d)
    result=_op.activation(result,activation=_op.Activation.softmax)
    print('-'*50,'Result Node Name','-'*50)
    print(result.name)
    print('-'*120)
    return result

