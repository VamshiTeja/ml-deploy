from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys
sys.path.append("../")
sys.path.append("../model")
import numpy as np

import tensorflow as tf 
import tensorflow.contrib.layers as layers
import pickle

from model.SimpleModel import SimpleModel
from utils import load_batched_data, dotdict
from config import cfg, cfg_from_file
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

train_datapath = "preprocess/train.pickle"
checkpoint_dir = "checkpoints/"
class Inference:
    def __init__(self):
        # batch size is 1 for prediction
        cfg.batch_size = 1
        self.model = SimpleModel(cfg)
        print("[INFO]: Building Graph ...")
        self.model.build_graph()
        self.sess = tf.InteractiveSession(graph=self.model.graph)
        self.sess.run(self.model.initial_op)
        self.model.saver.restore(self.sess, tf.train.latest_checkpoint(checkpoint_dir))
    
    def make_prediction(self, test_tuple):
        pred_price, loss = self.sess.run([self.model.pred_price, self.model.loss], 
                    feed_dict={self.model.name: test_tuple[0], self.model.item_condition_id:test_tuple[1], self.model.category_id:test_tuple[2], 
                        self.model.brand_id:test_tuple[3], self.model.shipping:test_tuple[4], self.model.item_description:test_tuple[5], 
                        self.model.target_price:test_tuple[-1]})
        return pred_price, loss

def load_data(datapath, is_Train=True):
        with open(datapath, 'rb') as handle:
            data = pickle.load(handle)
        
        names = data['name']
        item_condition_id = data['item_condition_id']
        category_name = data['category_name']
        brand_name    = data['brand_name']
        shipping     = data['shipping']
        item_description = data['item_description']

        if(is_Train):
            price = data['price']
            return (names, item_condition_id, category_name, brand_name, shipping, item_description, price)
        else:
            return (names, item_condition_id, category_name, brand_name, shipping, item_description)

if __name__ == '__main__':

    tester = Inference()
    test_data = load_data("preprocess/train.pickle")
    test_data = next(load_batched_data(test_data,1,cfg))
    pred_price, _ = tester.make_prediction(test_data)
    print(pred_price)