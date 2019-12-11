from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys
sys.path.append("../")
sys.path.append("../model")
import numpy as np

import tensorflow as tf 
import tensorflow_hub as hub
import pickle

from model.SimpleModel import SimpleModel
from utils import load_batched_data, load_data
from config import cfg, cfg_from_file

train_datapath = "preprocess/train.pickle"
checkpoint_dir = "checkpoints/"
class Inference:
    def __init__(self):
        # batch size is 1 for prediction
        cfg.batch_size = 1
        self.model = SimpleModel(cfg)
        self.model.build_graph()
        self.sess = tf.InteractiveSession(graph=self.model.graph)
        self.sess.run(self.model.initial_op)
        self.model.saver.restore(self.sess, tf.train.latest_checkpoint(checkpoint_dir))
    
    def make_prediction(self, test_tuple):
        pred_price = self.sess.run([self.model.pred_price], 
                    feed_dict={self.model.name: test_tuple[0], self.model.item_condition_id:test_tuple[1], self.model.category_id:test_tuple[2], 
                        self.model.brand_id:test_tuple[3], self.model.shipping:test_tuple[4], self.model.item_description:test_tuple[5] 
                        })
        return pred_price[0][0][0]

class DataProcessor:
    def __init__(self, cat_encoding, brand_encoding):
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inp = tf.placeholder(dtype=tf.string, shape=[None])
            self.embed = hub.Module(module_url)
            self.embedded_text = self.embed(self.inp)
            init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
        self.graph.finalize()
        self.session = tf.Session(graph=self.graph)
        self.session.run(init_op)

        with open(cat_encoding, 'rb') as handle:
            self.le_cat = pickle.load(handle)

        with open(cat_encoding, 'rb') as handle:
            self.le_brand = pickle.load(handle)

    def process(self,data):
        cat_name = np.array(self.le_cat.transform([data[2]]))
        brand_name = np.array(self.le_brand.transform([data[3]]))
        to_be_embedded_data = [data[0], data[5]]
        name, item_description = self.session.run(self.embedded_text, feed_dict={self.inp: to_be_embedded_data})
        name, item_description = np.expand_dims(name, axis=0), np.expand_dims(item_description, axis=0)
        return (name,np.expand_dims([data[1]],axis=0),cat_name, brand_name, np.expand_dims([data[4]],axis=0), item_description)


if __name__ == '__main__':
    tester = Inference()
    name = "MLB Cincinnati Reds T Shirt Size XL"
    item_cond = "3"
    cat_name = "Men/Tops/T-shirts"
    brand_name = "missing"
    shipping = "1"
    item_desc = "No description yet"
    data = (name, item_cond, cat_name, brand_name, shipping, item_desc)
    pro = DataProcessor(cfg.category_model_dir, cfg.brand_model_dir)
    test_data = pro.process(data)
    pred_price = tester.make_prediction(test_data)
    print(pred_price)


