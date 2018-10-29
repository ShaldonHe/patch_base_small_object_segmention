import tensorflow as tf
from tensorflow import nn as nn
from tensorflow.python.training import training
from model.Nets import patch_generator as g_patch
from model.Nets import patch_discriminator as d_patch
from model.Nets import patch_multi_task as multi_patch
from . import config as _c
_layer_c=_c.parse_args()

def patch_gan_fn():
    pass


def patch_segmentation_fn(features, labels, mode, params):
    predict = s_patch(features["images"], mode == tf.estimator.ModeKeys.TRAIN)
    
    seq_result=predict['seq_result']
    class_result=predict['class_result']
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])

    def iou_loss(label,seq_result):
        inter=tf.reduce_sum(tf.multiply(label,predict))+1e-8
        union=tf.reduce_sum(tf.subtract(tf.add(label,predict),tf.multiply(label,predict)))+1e-8
        loss=tf.subtract(tf.constant(1.0, dtype=tf.float32),tf.divide(inter,union))
        return loss

    def class_loss(label,class_result):
        return tf.losses.mean_squared_error(tf.reduce_sum(label,axis=(1,2,3)),class_result)


    if labels is not None:
        # loss = iou_loss(labels,predict)
        # loss = tf.losses.mean_squared_error(labels,predict)#iou_loss(labels,predict)
        # loss = tf.reduce_sum(tf.sqrt( tf.abs(labels-predict)))
        loss = class_loss
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



def patch_multitask_fn(features, labels, mode, params):
    predict = s_patch(features["images"], mode == tf.estimator.ModeKeys.TRAIN)
    
    seq_result=predict['seq_result']
    class_result=predict['class_result']
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])

    def iou_loss(label,seq_result):
        inter=tf.reduce_sum(tf.multiply(label,predict))+1e-8
        union=tf.reduce_sum(tf.subtract(tf.add(label,predict),tf.multiply(label,predict)))+1e-8
        loss=tf.subtract(tf.constant(1.0, dtype=tf.float32),tf.divide(inter,union))
        return loss

    def class_loss(label,class_result):
        return tf.losses.mean_squared_error(tf.reduce_sum(label,axis=(1,2,3)),class_result)


    if labels is not None:
        # loss = iou_loss(labels,predict)
        # loss = tf.losses.mean_squared_error(labels,predict)#iou_loss(labels,predict)
        # loss = tf.reduce_sum(tf.sqrt( tf.abs(labels-predict)))
        loss = class_loss
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
