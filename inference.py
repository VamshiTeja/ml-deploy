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
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


#define hyperparameters
batch_size = 1
learning_rate = 2e-4
num_epochs = 100
num_iters = 10000
is_Train = 1
grad_clip = 1
num_units = 512
sent2vecdim = 600
num_categories = 1354
cat_dim = 300
num_brands = 6312
brand_dim = 300
std = 0.02
restore = 0
mode = "train"
dataset = "mercari-price-prediction"

train_datapath = "preprocess/train.pickle"
checkpoint_dir = "checkpoints/"


class Inference:
    def __init__(self):
        args_dict = self._default_configs()
        self.args = dotdict(args_dict)
        self.model = SimpleModel(self.args)
        print("[INFO]: Building Graph ...")
        self.model.build_graph()
        self.sess = tf.InteractiveSession(graph=self.model.graph)
        self.sess.run(self.model.initial_op)
        self.model.saver.restore(self.sess, tf.train.latest_checkpoint(checkpoint_dir))

    def _default_configs(self):
        return {
            'batch_size': batch_size,
            'is_Train' : is_Train,
			'learning_rate': learning_rate,
			'num_epochs': num_epochs,
            'num_iters': num_iters,
			'grad_clip': grad_clip,
            'num_units': num_units,
			'sent2vecdim': sent2vecdim,
            'num_categories': num_categories,
			'cat_dim': cat_dim,
            'num_brands': num_brands,
            'brand_dim' : brand_dim,
			'mode': mode,
			'std': std,
			'restore': restore,
			'dataset': dataset
			}
    
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
    test_data = next(load_batched_data(test_data,tester.args.batch_size,tester.args))
    pred_price, _ = tester.make_prediction(test_data)
    print(pred_price)