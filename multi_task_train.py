import tensorflow as tf
import numpy as np
import libs.common.class_interface as libci
from libs.collection.prep_utility import imread,imwrite
import matplotlib.pyplot as plt
import train.config as _train_con
args=_train_con.parse_args()
run_config = tf.ConfigProto(log_device_placement=True,allow_soft_placement=False,device_count={'gpu':1})
X = tf.placeholder("float", [None]+args.data_shape[:], name="X")
Y = tf.placeholder("float", [None]+ args.label_shape[:], name="Y")
predict=patch_segmentation(X)
import libs.model.losses as libss
loss=libss.iou_loss(Y,predict)
train_op=tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(loss=loss)
with tf.Session(config=run_config) as sess:
    sess.run(tf.global_variables_initializer())
    for iters in range(len(data)):
        plt.imshow(data[iters])
        plt.show()
        plt.imshow(label[iters])
        plt.show()
        _,r_loss,r_predict=session.run([train_op, loss , predict],{X: data[iters],Y: label[iters]})
        print(r_loss)
        print(r_predict.shape)