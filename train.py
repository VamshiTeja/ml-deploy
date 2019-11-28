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
from utils import load_batched_data
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


#define hyperparameters
batch_size = 32
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

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def make_str(config):
	'''
		returns str representation of a dict
	'''
	config_str = ""
	for k,v in config.iteritems():
		config_str += str(k)+":"+str(v)+"_"
	return config_str

# TODO Read Data
class Trainer:
    def __init__(self):
        self.args = None

    def load_data(self,datapath, is_Train=True):
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

    def train(self):
        args_dict = self._default_configs()
        args = dotdict(args_dict)
        data = self.load_data(train_datapath)

        print("[INFO]: Data Loaded")
        model = SimpleModel(args)
        print("[InFO]: Building Graph ...")
        model.build_graph()

        num_batches = int(len(data[0])/args.batch_size)
        print("[Info]: Number of batches are %d " %num_batches)

        with tf.Session(graph=model.graph) as sess:
            writer = tf.summary.FileWriter("logging",graph=model.graph)

            if(args.restore==1):
                model.saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
            else:
                sess.run(model.initial_op)

            for iteration in range(num_iters):
                X_tuple = next(load_batched_data(data,args.batch_size,args))
                _, loss, summary = sess.run([model.opt, model.loss, model.summary_op],
                                            feed_dict={model.name: X_tuple[0], model.item_condition_id:X_tuple[1], model.category_id:X_tuple[2], 
                                            model.brand_id:X_tuple[3], model.shipping:X_tuple[4], model.item_description:X_tuple[5], model.target_price:X_tuple[-1]})

                writer.add_summary(summary, iteration)
                if(iteration!=0 & iteration%50==0):
                    print("[Info]: Iter:%d/%d , loss: %f "%(iteration+1, num_iters, loss))

                if(iteration%100==0):
                    model.saver.save(sess, checkpoint_dir+args.dataset+"_SimpleModel.ckpt", global_step = model.global_step)

                # Test after every 2 epochs
                i=0
                if(iteration%100 == 0):
                    test_tuple = next(load_batched_data(data,args.batch_size,args))

                    _, loss = sess.run([model.pred_price, model.loss],
                                            feed_dict={model.name: test_tuple[0], model.item_condition_id:test_tuple[1], model.category_id:test_tuple[2], 
                                            model.brand_id:test_tuple[3], model.shipping:test_tuple[4], model.item_description:test_tuple[5], model.target_price:test_tuple[-1]})
                    # rmse = np.sqrt(np.mean(pred_price-))

                    print("Test RMSE: %f"%loss)
					

if __name__ == '__main__':
	runner = Trainer()
	runner.train()