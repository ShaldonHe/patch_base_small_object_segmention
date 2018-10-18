import numpy as np
import tensorflow as tf
import train.run as train
import prepare.prepare as prep
tf.logging.set_verbosity(tf.logging.INFO)
prep.progress()
train.progress()