import tensorflow as tf
import numpy as np
import libs.common.class_interface as libci
from matplotlib.pyplot import *
from libs.collection.prep_utility import imread,imwrite
def parse_args():
    desc = "data config"
    parser = {}
    im_size=512
    im_c=3
    output_c=1
    pyramids=[1]
    
    parser['origin_image_dir'] = '/home/sheldon/Documents/Data/CHXD/segmentation/Train/Images/Enhanced'
    parser['origin_label_dir'] = '/home/sheldon/Documents/Data/CHXD/segmentation/Train/Labels/Cropped'

    parser['model_image_dir'] = './data/model/data/prep_image'
    parser['model_label_dir'] = './data/model/data/prep_label'

    parser['data_ext'] ='png|jpeg|bmp|jpg'

    parser['data_info_file'] = './data/model/data/data_layer_info.npz'
    
    parser['radius'] = im_size//2
    parser['pyramids'] = pyramids
    parser['stride'] = max(im_size//8,8)

    parser['patch_shape'] = [im_size,im_size]
    
    parser['data_shape'] = [im_size,im_size,im_c]
    parser['label_shape'] = [im_size,im_size,output_c]
    parser['layer_num'] = output_c
    parser['train_batch_size'] = 1
    return libci.Dic2Object(parser)
    
#     return libci.check_config(parser.parse_args())

args=parse_args()

import libs.common.data_interface as libdi
def MA_segmention_debug_data():
    c=args
    data_info=np.load(c.data_info_file)['data_info'].item()

    def _transform_image_data(file_path):
        img=imread(file_path)
        img=img.astype(np.float)/255.0
        return img
    
    data_dict=data_info['patch_image_dict']
    keys=list(data_dict.keys())[:40]
    _data_dict={}
    for k in keys:
        _data_dict[k]=data_dict[k]
    data=libdi.rank_file_reader(_data_dict,block_shape=c.data_shape,transform_fn=_transform_image_data)
 
    def _transform_label_data(file_path):
        img=imread(file_path,0)
        if img.max()>1:
            img=img.astype(np.float)/255.0
            
        label_img=np.ones(shape=(img.shape[0],img.shape[1],c.layer_num))
        if c.layer_num==1:
            label_img[:,:,0]=img.astype(np.float)
        else:
            for i in range (c.layer_num):
                label_img[:,:,i]=(img==i).astype(np.float)
        return label_img
    label_dict=data_info['patch_label_dict']
    _label_dict={}
    for k in keys:
        _label_dict[k]=label_dict[k]

    label=libdi.rank_file_reader(_label_dict,block_shape=c.label_shape,transform_fn=_transform_label_data)
    return data,label

data,label=MA_segmention_debug_data()
for i in range(1,10):
    subplot(3,3,i)
    imshow(data[i][0])
show()

for i in range(1,10):
    subplot(3,3,i)
    imshow(label[i][0][:,:,0])
show()


import libs.model.blocks as _bk
import libs.model.layers as _l
import libs.model.ops as _op
def patch_segmentation(x):
    # input data shape
    # 256*256*(3:origin image )
    print('-'*50,'Input Node Name','-'*50)
    print(x.name)
    print('-'*120)
    k_size=3
    init_filter=32
    model_parms=dict(kernel_size=(k_size,k_size),use_bias=False,conv=_l.Conv.conv2d,norm=_l.Norm.batch,activation=_op.Activation.relu)
    block_0=_bk.conv_norm_activation(x,filters=init_filter,**model_parms)
    block_1=_bk.res_v1_block(block_0,**model_parms) 
    block_2=_bk.res_v1_block(block_1,**model_parms) 
    block_3=_bk.res_v1_block(block_2,**model_parms) 
    block_4=_bk.res_v1_block(block_3,**model_parms) 
    unet_conv_t_4=_bk.ushape_block(block_4,block_3,img_size=block_3.shape[1:3].as_list(),filters=block_3.shape[3].value,**model_parms)
    unet_conv_t_3=_bk.ushape_block(unet_conv_t_4,block_2,img_size=block_2.shape[1:3].as_list(),filters=block_2.shape[3].value,**model_parms)
    unet_conv_t_2=_bk.ushape_block(unet_conv_t_3,block_1,img_size=block_1.shape[1:3].as_list(),filters=block_1.shape[3].value,**model_parms)
    unet_conv_t_1=_bk.ushape_block(unet_conv_t_2,block_0,img_size=block_0.shape[1:3].as_list(),filters=block_0.shape[3].value,**model_parms)
    result=_bk.conv_norm_activation(unet_conv_t_1,filters=args.layer_num,kernel_size=(1,1),activation=_op.Activation.relu)
    print('-'*50,'Result Node Name','-'*50)
    print(result.name)
    print('-'*120)
    return result


import libs.model.losses as libss
# with tf.device('/gpu:0'):
#     X = tf.placeholder("float", [None]+args.data_shape[:], name="X")
#     Y = tf.placeholder("float", [None]+ args.label_shape[:], name="Y")
#     predict=patch_segmentation(X)
#     loss=libss.iou_loss(Y,predict)
#     loss=tf.reduce_sum(Y)+tf.reduce_sum(predict)

X = tf.placeholder("float", [None]+args.data_shape[:], name="X")
Y = tf.placeholder("float", [None]+ args.label_shape[:], name="Y")
predict=patch_segmentation(X)
loss=libss.iou_loss(Y,predict)
loss=tf.reduce_sum(Y)+tf.reduce_sum(predict)

train_op=tf.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(loss=loss)

run_config = tf.ConfigProto(log_device_placement=True,allow_soft_placement=False)
with tf.Session(config=run_config) as sess:
    sess.run(tf.global_variables_initializer())
    for iters in range(len(data)):
        # plt.imshow(data[iters][0])
        # plt.show()
        # plt.imshow(label[iters][0][:,:,0])
        # plt.show()
        _,r_loss,r_predict=sess.run([train_op, loss , predict],{X: data[iters],Y: label[iters]})
        _r=r_predict[0][:,:,0]
        _l=label[iters][0,:,:,0]
        print(r_loss)
        # print(r_predict.shape)