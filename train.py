from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys
sys.path.append("../")
sys.path.append("../model")
import numpy as np

import tensorflow as tf 
import pickle

from model.SimpleModel import SimpleModel
from utils import load_batched_data, load_data
from config import cfg, cfg_from_file
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

train_datapath = "preprocess/train.pickle"
checkpoint_dir = "checkpoints/"

class Trainer:
    def __init__(self):
        self.cfg = None

    def train(self):
        data = load_data(train_datapath)

        print("[INFO]: Data Loaded")
        model = SimpleModel(cfg)
        print("[INFO]: Building Graph ...")
        model.build_graph()

        num_batches = int(len(data[0])/cfg.batch_size)
        print("[INFO]: Number of batches are %d " %num_batches)

        with tf.Session(graph=model.graph) as sess:
            writer = tf.summary.FileWriter("logging",graph=model.graph)

            if(cfg.restore==1):
                model.saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
            else:
                sess.run(model.initial_op)

            for iteration in range(cfg.num_iters):
                X_tuple = next(load_batched_data(data,cfg.batch_size,cfg))
                _, loss, summary = sess.run([model.opt, model.loss, model.summary_op],
                                            feed_dict={model.name: X_tuple[0], model.item_condition_id:X_tuple[1], model.category_id:X_tuple[2], 
                                            model.brand_id:X_tuple[3], model.shipping:X_tuple[4], model.item_description:X_tuple[5], model.target_price:X_tuple[-1]})

                writer.add_summary(summary, iteration)
                if(iteration!=0 & iteration%50==0):
                    print("[Info]: Iter:%d/%d , loss: %f "%(iteration+1, cfg.num_iters, loss))

                if(iteration%100==0):
                    model.saver.save(sess, checkpoint_dir+cfg.dataset+"_SimpleModel.ckpt", global_step = model.global_step)

                # Test after every 2 epochs
                if(iteration%100 == 0):
                    test_tuple = next(load_batched_data(data,cfg.batch_size,cfg))

                    _, loss = sess.run([model.pred_price, model.loss],
                                            feed_dict={model.name: test_tuple[0], model.item_condition_id:test_tuple[1], model.category_id:test_tuple[2], 
                                            model.brand_id:test_tuple[3], model.shipping:test_tuple[4], model.item_description:test_tuple[5], model.target_price:test_tuple[-1]})
                    print("Test RMSE: %f"%loss)
					

if __name__ == '__main__':
	runner = Trainer()
	runner.train()