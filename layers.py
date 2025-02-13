# Custom L1 Distance layer module 


# Import dependencies
import tensorflow as tf
from tensorflow.keras.layers import Layer

# Custom L1 Distance Layer from Jupyter 
class L1Dist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super(L1Dist,self).__init__(**kwargs)
       
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
         
         input_embedding = tf.convert_to_tensor(input_embedding, dtype=tf.float32)
         validation_embedding = tf.convert_to_tensor(validation_embedding, dtype=tf.float32)
    
         return tf.math.abs(input_embedding - validation_embedding)