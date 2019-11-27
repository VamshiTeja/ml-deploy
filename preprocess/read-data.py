import os, sys
import pandas as pd 
import sklearn
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn import preprocessing
import pickle

import sent2vec

TRAIN_DIR = "../data/train.tsv"
TEST_DIR  = "../data/test_stg2.tsv"

cols = ["train_id", "name", "item_condition_id", "category_name", "brand_name", "price", "shipping", "item_description"]
test_cols = ["train_id", "name", "item_condition_id", "category_name", "brand_name", "shipping", "item_description"]

print("Loading Training Data ...")
train_data = pd.read_csv(TRAIN_DIR, sep="\t",encoding = "utf-8", names=cols, skiprows=1)
print("Loading Testing Data ...")
test_data = pd.read_csv(TEST_DIR, sep="\t",encoding = "utf-8", names=test_cols, skiprows=1)
# print (train_data.isnull().any())
# print(test_data.isnull().any())

#fill missing values
def fill_missing_data(data):
    data.category_name.fillna(value = "missing", inplace = True)
    data.item_description.fillna(value = "missing", inplace = True)
    data.brand_name.fillna(value="missing", inplace=True)
    return data

print("Handling missing values by marking missing ...")
train_data = fill_missing_data(train_data)
test_data  = fill_missing_data(test_data)

# Encode categorical variables
le_cat = preprocessing.LabelEncoder()
le_cat.fit(list(np.unique(list(train_data.category_name)+ list(test_data.category_name))))
# train_data.category_name = le_cat.transform(train_data.category_name)
# test_data.category_name = le_cat.transform(test_data.category_name)
# print("Category Labels Encoded!")

le_brand = preprocessing.LabelEncoder()
le_brand.fit(list(np.unique(list(train_data.brand_name)+ list(test_data.brand_name))))
# train_data.brand_name = le_brand.transform(train_data.brand_name)
# test_data.brand_name = le_brand.transform(test_data.brand_name)
# print("Brand Labels Encoded!")

#make name and item description as seq of words
# vocab = []

# def split_list(list, v):
#     for x in list:
#         v = v + x.split(" ")
#     return v

# vocab_list =  list(train_data.name) + list(test_data.name) + list(train_data.item_description) + list(test_data.item_description)

# vocab = split_list(vocab_list, vocab)
# vocab = np.unique(vocab)
# print(vocab)

# vocab_dict = dict(zip(vocab_list, range(len(vocab))))
# vocab_inverse_dict = dict(zip(range(len(vocab), vocab_list)))

print("Loading Sent2vec Model ...")
model = sent2vec.Sent2vecModel()
model.load_model('../sent2vec/wiki_unigrams.bin')
emb = model.embed_sentence("once upon a time .") 

# print("Converting names to embeddings ...")
# embs = model.embed_sentences(list(train_data.name))

# print("converting item description to embeddings ...")
# embs = model.embed_sentences(list(train_data.item_description))
# print(embs.shape)


def get_data(datadf, model, le_cat, le_brand, is_train=True):
    name_embs = model.embed_sentences(list(datadf.name))
    des_embs = model.embed_sentences(list(datadf.item_description))
    if(is_train):
        X = {
            'name' : name_embs,
            'item_condition_id' : np.array(datadf.item_condition_id),
            'category_name' : np.array(le_cat.transform(datadf.category_name)),
            'brand_name'  : np.array(le_brand.transform(datadf.brand_name)),
            'price' : np.array(datadf.price),
            'shipping' : np.array(datadf.shipping),
            'item_description' : des_embs
        }
    else:
        X = {
            'name' : name_embs,
            'item_condition_id' : np.array(datadf.item_condition_id),
            'category_name' : np.array(le_cat.transform(datadf.category_name)),
            'brand_name'  : np.array(le_brand.transform(datadf.brand_name)),
            'shipping' : np.array(datadf.shipping),
            'item_description' : des_embs
        }
    return X

print("Preparing Training Data ...")
train = get_data(train_data, model, le_cat, le_brand )
test = get_data(test_data, model, le_cat, le_brand,is_train=False)

with open('train.pickle', 'wb') as handle:
    pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('test.pickle', 'wb') as handle:
    pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('filename.pickle', 'rb') as handle:
#     b = pickle.load(handle)