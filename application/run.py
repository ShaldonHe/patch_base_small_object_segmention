import numpy as np
import application.config as con
from matplotlib.pyplot import *
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
import libs.libs.file_interface as libfi
from libs.collection.prep_utility import imread,imwrite
import libs.lib_model.models as model
import libs.libs.view_interface as libvi
from matplotlib.pyplot import *
import os.path as osp
args=con.parse_layer_args()
from matplotlib.pyplot import *
file_list=libfi.getfilesbyext(folder=args.image_dir,ext=args.image_ext)


def _get_dist_img(shape,center=None):
    if center is None:
        center=(shape[0]//2,shape[1]//2)
    x,y=np.meshgrid(range(0,shape[1]),range(0,shape[0]))
    x=x-center[0]
    y=y-center[1]
    dist_im=np.sqrt(x**2+y**2)
    return dist_im



weight_mask=_get_dist_img(shape=con._model_layer.output_shape[:2])
weight_mask=1/(weight_mask/weight_mask.max()+0.5)

for f in file_list:
    image=imread(f)
    image=image.astype(np.float)/255.0
    nn=model.SegmentionModel()
    nn.load_from_checkpoint_dir(osp.abspath('./data/model/layer'))
    nn.init_model('random_shuffle_queue_DequeueMany:1','softmax2d/truediv:0')
    im_size=nn.input_shape[1]
    ids,_=libvi.image_to_blocksinfo(image,radius=im_size//2,stride=im_size//4,pyramid=[1])
    input_image=[]
    for i in ids:
        input_image.append(libvi.get_block_fromids(image=image,ids=i,block_size=[im_size,im_size]))
    shape=list(image.shape[:2])
    shape.append(con._model_layer.layer_num)
    mask=np.zeros(shape=shape)

    for num,_ids in enumerate(ids):
        in_data=np.array(input_image[num:num+args.batch_size])
        result=nn.predict(in_data)
        result=result[0]

        for i in range(1,result.shape[2]):
            result[:,:,i]=result[:,:,i]*weight_mask
        # for i in range(1,result.shape[3]):
        #     result[:,:,i]=result[:,:,i]*i
        # mean_result=np.sum(result)
        # arg_result=np.argmax(result,axis=-1)
        mask=libvi.set_block(mask,result,_ids.centroid,_ids.radius)
    subplot(1,2,1)
    imshow(image)
    subplot(1,2,2)
    imshow(np.argmax(mask,axis=-1))
    show()