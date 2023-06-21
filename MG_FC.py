from final_utils import fc_model_softmax, model_inceptionresnet_multigap
import tensorflow as tf
import numpy as np
from keras.layers import Concatenate

model_fc = fc_model_softmax(input_num=16928)
model_fc.load_weights('models/Softmax/26.08_scheduler_tests/model_fc_softmax_26_08_128bs_epoch15.hdf5')
model_base=model_inceptionresnet_multigap()

class model_multigap_fc(tf.keras.Model):
    def __init__(self, model_base=model_base, model_fc=model_fc):
        super(model_multigap_fc, self).__init__()
        
        self.layer_multigap = model_base
        self.layer_fc = model_fc

    def call(self, img):
        x = self.layer_multigap(img)
        x = self.layer_fc(x)
            
        return x