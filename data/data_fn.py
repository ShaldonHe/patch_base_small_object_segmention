from . import config as con
import libs.common.file_interface as libfi
import numpy as np
import tensorflow as tf
import libs.common.data_interface as libsdi
from libs.collection.prep_utility import imread,imwrite
def layer_segmention_input(input_dicts):
    c=con.parse_layer_args()
    data_info=np.load(c.data_info)['layer_dict'].item()
    data_dict={}
    for key in data_info:
        item=data_info[key]
        data_dict[key]=item.img

    def _transform_image_data(file_path):
        img=imread(file_path)
        img=img.astype(np.float)/255.0
        return img

    data=libsdi.mutil_image_reader(data_dict,args=c,block_shape=c.data_shape,transform_fn=_transform_image_data)
    if libfi.exist('./data/model_data/e_loss.npz'):
        loaded=np.load('./data/model_data/e_loss.npz')
        e_loss=loaded['e_loss']
        ids=loaded['ids']
        data.read_cache_info(e_loss=e_loss,ids=ids)




    label_dict=libfi.getfiledicbyext(c.label_path,'png')
    c.pyramid=[1]

    def _transform_image_label(file_path):
        img=imread(file_path,0)
        layer_num=float(c.layer_num)
        label_img=np.ones(shape=(img.shape[0],img.shape[1],c.layer_num))
        for i in range (c.layer_num):
            label_img[:,:,i]=(img==i).astype(np.float)
        return label_img

    label=libsdi.mutil_image_reader(label_dict,args=c,block_shape=c.label_shape,transform_fn=_transform_image_label)
    return tf.estimator.inputs.numpy_input_fn(
    x={"images": data},
    y=label,
    batch_size=c.train_batch_size,
    num_epochs=None,
    shuffle=True)





def test_input_fn_layer_segmention():
    c=con.parse_layer_args()
    data_info=np.load(c.data_info)['layer_dict'].item()
    data_dict={}
    for key in data_info:
        item=data_info[key]
        data_dict[key]=item.img

    def _transform_image_data(file_path):
        img=imread(file_path)
        img=img.astype(np.float)/255.0
        return img

    data=libsdi.mutil_image_reader(data_dict,args=c,block_shape=c.data_shape,transform_fn=_transform_image_data)
    return tf.estimator.inputs.numpy_input_fn(
    x={"images": data},
    batch_size=c.test_batch_size,
    num_epochs=None,
    shuffle=False),data

def train_input_fn_layer_segmention():
    c=con.parse_layer_args()
    data_info=np.load(c.data_info)['layer_dict'].item()
    data_dict={}
    for key in data_info:
        item=data_info[key]
        data_dict[key]=item.img

    def _transform_image_data(file_path):
        img=imread(file_path)
        img=img.astype(np.float)/255.0
        return img
    data=libsdi.mutil_image_reader(data_dict,args=c,block_shape=c.data_shape,transform_fn=_transform_image_data)    


    # data=libsdi.mutil_image_reader(data_dict,args=c,block_shape=c.data_shape,transform_fn=_transform_image_data)
    label_dict=libfi.getfiledicbyext(c.label_path,'png')
    c.pyramid=[1]
    def _transform_image_label(file_path):
        img=imread(file_path,0)
        layer_num=float(c.layer_num)
        label_img=np.ones(shape=(img.shape[0],img.shape[1],c.layer_num))
        for i in range (c.layer_num):
            label_img[:,:,i]=(img==i).astype(np.float)
        # print()
        return label_img
    label=libsdi.mutil_image_reader(label_dict,args=c,block_shape=c.label_shape,transform_fn=_transform_image_label)
    return tf.estimator.inputs.numpy_input_fn(
    x={"images": data},
    y=label,
    batch_size=c.train_batch_size,
    num_epochs=None,
    shuffle=True)