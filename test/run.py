import train.config as con
import model.model_fn as model_fn
import data.data_fn as data_input
from matplotlib.pyplot import *
import numpy as np
import tensorflow as tf
from matplotlib.pyplot import *
tf.logging.set_verbosity(tf.logging.INFO)
def progress():
    args=con.parse_layer_args()
    model_params = {"learning_rate": args.learning_rate,'model_dir':args.model_dir}
    nn = tf.estimator.Estimator(model_fn=model_fn.Layer_Segmention_fn,model_dir=args.model_dir,params=model_params)
    test_input_fn = data_input.test_input_fn_layer_segmention()
    result=nn.predict(input_fn=test_input_fn)
    
    for num,r in enumerate(result):
        print(num,set(r['result'].flatten()))
        print(num)
        # print(i['predict'])

    print(result)