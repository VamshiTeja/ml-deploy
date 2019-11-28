from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf 
import tensorflow.contrib.layers as layers

import os,sys

class SimpleModel:
    def __init__(self,args):
        self.args = args

    def model(self,name,item_desc, item_cond_id, cat_name, brand_name, shipping):
        with tf.variable_scope("model"):
            name = tf.layers.dense(name, units=self.args.num_units,activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            item_desc = tf.layers.dense(item_desc, units=self.args.num_units,activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            item_cond_id = tf.layers.dense(item_cond_id, units=self.args.num_units,activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            cat_name = tf.layers.dense(cat_name, units=self.args.num_units,activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            brand_name = tf.layers.dense(brand_name, units=self.args.num_units,activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            shipping = tf.layers.dense(shipping, units=self.args.num_units,activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

            concat = tf.concat([name,item_desc, item_cond_id, cat_name, brand_name, shipping],1)

            fc1 = tf.layers.dense(concat, units=1024, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            fc2 = tf.layers.dense(fc1, units=512, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

            res = tf.layers.dense(fc2, units=1, activation=None, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        return res

    def build_graph(self):

        self.graph = tf.Graph()
        with(self.graph.as_default()):
            self.name = tf.placeholder(dtype=tf.float32, shape=(self.args.batch_size,self.args.sent2vecdim))
            self.item_description = tf.placeholder(dtype=tf.float32, shape=(self.args.batch_size,self.args.sent2vecdim))
            self.item_condition_id = tf.placeholder(dtype=tf.float32, shape=(self.args.batch_size,1))
            self.shipping = tf.placeholder(dtype=tf.float32, shape=(self.args.batch_size,1))

            self.target_price = tf.placeholder(dtype=tf.float32, shape=(self.args.batch_size, 1))

            # category embeddings
            self.category_id = tf.placeholder(dtype=tf.int32, shape=(self.args.batch_size))
            self.cat_embedding = tf.Variable(tf.random_uniform([self.args.num_categories, self.args.cat_dim], -1, 1))
            self.category = tf.nn.embedding_lookup(self.cat_embedding, self.category_id)

            # brand embeddings
            self.brand_id = tf.placeholder(dtype=tf.int32, shape=(self.args.batch_size))
            self.brand_embedding = tf.Variable(tf.random_uniform([self.args.num_brands, self.args.brand_dim], -1, 1))
            self.brand = tf.nn.embedding_lookup(self.brand_embedding, self.brand_id)

            self.pred_price = self.model(self.name, self.item_description, self.item_condition_id, self.category, self.brand, self.shipping)

            theta  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="model")

            self.loss = tf.sqrt(tf.reduce_mean(tf.square(self.pred_price-self.target_price)))
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            if self.args.grad_clip==-1:
                self.opt = tf.train.AdamOptimizer(self.args.learning_rate,beta1=0.5).minimize(self.loss)
            else:
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, theta), 5)
                opti = tf.train.AdamOptimizer(self.args.learning_rate,beta1=0.5)
                self.opt= opti.apply_gradients(zip(grads, theta), global_step=self.global_step)

            tf.summary.scalar('loss',self.loss)  

            self.initial_op = tf.global_variables_initializer()
            self.summary_op = tf.summary.merge_all()
            self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=5, keep_checkpoint_every_n_hours=1)