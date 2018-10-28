import numpy as np
import tensorflow as tf
import sys
sys.path.append('./')
import train.run as train
import prepare.prepare as prep
tf.logging.set_verbosity(tf.logging.INFO)
prep.progress()
train.activate_progress()
# train.norm_progress()