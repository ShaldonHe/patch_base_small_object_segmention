import tensorflow as tf
from tensorflow import nn as nn
from tensorflow.python.training import training
from model.Nets import patch_generator as g_patch
from model.Nets import patch_discriminator as d_patch
from model.Nets import patch_segmentation as s_patch
from . import config as _c
from tensorflow.metrics import mean_iou as loss_iou
_c_d=_c.parse_d_args()
_c_g=_c.parse_g_args()



def patch_segmentation_fn(features, labels, mode, params):
    predict = s_patch(features["images"], mode == tf.estimator.ModeKeys.TRAIN)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])

    def iou_loss(label,predict):
        inter=tf.reduce_sum(tf.multiply(predict,label))
        union=tf.reduce_sum(tf.subtract(tf.add(predict,label),tf.multiply(predict,label)))+1e-8
        loss=tf.subtract(tf.constant(1.0, dtype=tf.float32),tf.divide(inter,union))
        return loss

    if labels is not None:
        tf.metrics.mean_iou
        loss = iou_loss(labels,predict)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    if mode == tf.estimator.ModeKeys.PREDICT:
        if labels is None:
            return tf.estimator.EstimatorSpec(mode=mode,predictions={"result": tf.argmax(predict, axis=3),'predict':predict,'data':features["images"]})
        else:        
            return tf.estimator.EstimatorSpec(mode=mode,predictions={"label":labels,'predict':predict,'data':features["images"],'loss':loss})
    
    eval_metric_ops = {'Accuracy':tf.metrics.accuracy(labels,predict),'label':labels,'predict':predict,'data':features['images']}
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode,loss=loss,eval_metric_ops=eval_metric_ops)

    return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)





def patch_generator_fn(features, labels, mode, params):
    predict = g_patch(features["images"], mode == tf.estimator.ModeKeys.TRAIN)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
    if labels is not None:
        loss = tf.losses.mean_squared_error(labels,predict)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    if mode == tf.estimator.ModeKeys.PREDICT:
        if labels is None:
            return tf.estimator.EstimatorSpec(mode=mode,predictions={"result": tf.argmax(predict, axis=3),'predict':predict,'data':features["images"]})
        else:        
            return tf.estimator.EstimatorSpec(mode=mode,predictions={"result": tf.argmax(predict, axis=3),'predict':predict,'data':features["images"],'loss':loss})
    


    eval_metric_ops = {"Accuracy": tf.metrics.accuracy(
        labels=tf.argmax(labels, axis=3),
        predictions=tf.argmax(predict, axis=3))}

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode,loss=loss,eval_metric_ops=eval_metric_ops)

    return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)

def patch_discriminator_fn(features, labels, mode, params):
    predict = d_patch(features["images"], mode == tf.estimator.ModeKeys.TRAIN)
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])

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

    return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)