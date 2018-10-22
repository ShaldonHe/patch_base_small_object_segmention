import tensorflow as tf
from tensorflow import nn as nn
from tensorflow.python.training import training
from model.Nets import patch_generator as g_patch
from model.Nets import patch_discriminator as d_patch
from . import config as _c
_c_d=_c.parse_d_args()
_c_g=_c.parse_g_args()

def patch_gan_fn():
    pass

def Patch_Generator_fn(features, labels, mode, params):
    predict = g_patch(features["images"], mode == tf.estimator.ModeKeys.TRAIN)
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
    #optimizer = tf.train.MomentumOptimizer(learning_rate=params['learning_rate'],momentum=0.9)

    if mode == tf.estimator.ModeKeys.PREDICT:
        if labels is None:
            return tf.estimator.EstimatorSpec(mode=mode,predictions={"result": tf.argmax(predict, axis=3),'predict':predict,'data':features["images"]})
        else:        
            return tf.estimator.EstimatorSpec(mode=mode,predictions={"result": tf.argmax(predict, axis=3),'predict':predict,'data':features["images"],'loss':loss})
    
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    eval_metric_ops = {"Accuracy": tf.metrics.accuracy(
            labels=tf.argmax(labels, axis=3),
            predictions=tf.argmax(predict, axis=3))}

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode,loss=loss,eval_metric_ops=eval_metric_ops)
    # run_config = tf.estimator.RunConfig()
    # tf.estimator.ModeKeys.TRAIN
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)