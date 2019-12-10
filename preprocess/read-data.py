import os, sys
import pandas as pd 
import sklearn
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn import preprocessing
import pickle
import tensorflow as tf
import tensorflow_hub as hub
import sent2vec

TRAIN_DIR = "../data/train.tsv"
TEST_DIR  = "../data/test_stg2.tsv"

cols = ["train_id", "name", "item_condition_id", "category_name", "brand_name", "price", "shipping", "item_description"]
test_cols = ["train_id", "name", "item_condition_id", "category_name", "brand_name", "shipping", "item_description"]

module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]

print("Loading Training Data ...")
train_data = pd.read_csv(TRAIN_DIR, sep="\t",encoding = "utf-8", names=cols, skiprows=1)
train_data = train_data[0:20000]
print("Loading Testing Data ...")
test_data = pd.read_csv(TEST_DIR, sep="\t",encoding = "utf-8", names=test_cols, skiprows=1)
test_data = test_data[0:20000]
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
print("number of unique categories: ", len(list(np.unique(list(train_data.category_name)+ list(test_data.category_name)))))
with open('category_encoding.pickle', 'wb') as handle:
    pickle.dump(le_cat, handle, protocol=pickle.HIGHEST_PROTOCOL)
# train_data.category_name = le_cat.transform(train_data.category_name)
# test_data.category_name = le_cat.transform(test_data.category_name)
# print("Category Labels Encoded!")

le_brand = preprocessing.LabelEncoder()
le_brand.fit(list(np.unique(list(train_data.brand_name)+ list(test_data.brand_name))))
print ("number of unique brands: ", len(list(np.unique(list(train_data.brand_name)+ list(test_data.brand_name)))))
with open('brand_encoding.pickle', 'wb') as handle:
    pickle.dump(le_brand, handle, protocol=pickle.HIGHEST_PROTOCOL)
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

# print("Loading Sent2vec Model ...")
# model = sent2vec.Sent2vecModel()
# model.load_model('../sent2vec/wiki_unigrams.bin')
# emb = model.embed_sentence("once upon a time .") 
train_data.fillna('missing', inplace=True)
embed = hub.Module(module_url)
desc_embeddings = []
name_embeddings = []
with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    for i in range(0, len(list(train_data.item_description)), 10000):
        desc_embedding = session.run(embed(list(train_data.item_description[i:i+10000])))
        desc_embeddings.append(desc_embedding.tolist())
    # desc_embedding = session.run(embed(list(train_data.item_description[i:])))
    # desc_embeddings.append(desc_embedding.tolist())

    for i in range(0, len(list(train_data.name)), 10000):
        name_embedding = session.run(embed(list(train_data.name[i:i+10000])))
        name_embeddings.append(name_embedding.tolist())
    # name_embedding = session.run(embed(list(train_data.name[i:])))
    # name_embeddings.append(name_embedding.tolist())
names = np.array(name_embeddings)
name_embeddings = np.reshape(name_embeddings, (20000, 512))

desc_embeddings = np.array(desc_embeddings)
desc_embeddings = np.reshape(desc_embeddings, (20000, 512))
print(name_embeddings)
print(len(desc_embeddings))
# print("Converting names to embeddings ...")
# embs = model.embed_sentences(list(train_data.name))

# print("converting item description to embeddings ...")
# embs = model.embed_sentences(list(train_data.item_description))
# print(embs.shape)


def get_data(datadf, le_cat, le_brand, is_train=True):
    # name_embs = model.embed_sentences(list(datadf.name))
    # des_embs = model.embed_sentences(list(datadf.item_description))
    if(is_train):
        X = {
            'name' : name_embeddings,
            'item_condition_id' : np.array(datadf.item_condition_id),
            'category_name' : np.array(le_cat.transform(datadf.category_name)),
            'brand_name'  : np.array(le_brand.transform(datadf.brand_name)),
            'price' : np.array(datadf.price),
            'shipping' : np.array(datadf.shipping),
            'item_description' : desc_embeddings
        }
    else:
        X = {
            'name' : name_embeddings,
            'item_condition_id' : np.array(datadf.item_condition_id),
            'category_name' : le_cat.transform(datadf.category_name),
            'brand_name'  : le_brand.transform(datadf.brand_name),
            'shipping' : np.array(datadf.shipping),
            'item_description' : desc_embeddings
        }
    return X

print("Preparing Training Data ...")
train = get_data(train_data, le_cat, le_brand )
# test = get_data(test_data, model, le_cat, le_brand,is_train=False)

with open('train.pickle', 'wb') as handle:
    pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('test.pickle', 'wb') as handle:
#     pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('filename.pickle', 'rb') as handle:
#     b = pickle.load(handle)