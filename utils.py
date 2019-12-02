import os,sys
import numpy as np
import pickle 

def load_batched_data(data,batch_size, cfg):
    '''
		Function to load data as batches
        Expects data to list of tuples
    '''
    num_samples = len(data[0])
    randIxs = np.random.permutation(num_samples)
    start,end =0,batch_size
    name, item_condition, category, brand, shipping, item_description, price = data

    while(end<=num_samples):
        batchInputs_name = np.zeros((batch_size,cfg.sent2vecdim))
        batchInputs_item_condition = np.zeros((batch_size,1))
        batchInputs_category = np.zeros((batch_size))
        batchInputs_brand = np.zeros((batch_size))
        batchInputs_shipping = np.zeros((batch_size,1))
        batchInputs_item_description = np.zeros((batch_size,cfg.sent2vecdim))
        batchInputs_price = np.zeros((batch_size,1))

        for batchI, origI in enumerate(randIxs[start:end]):
            batchInputs_name[batchI,:] = name[origI]				
            batchInputs_item_condition[batchI,:] = item_condition[origI]				
            batchInputs_category[batchI] = category[origI]				
            batchInputs_brand[batchI] = brand[origI]				
            batchInputs_shipping[batchI,:] = shipping[origI]				
            batchInputs_item_description[batchI,:] = item_description[origI]				
            batchInputs_price[batchI,:] = price[origI]				

            start += batch_size
            end += batch_size
            yield (batchInputs_name,batchInputs_item_condition,batchInputs_category,batchInputs_brand,batchInputs_shipping,batchInputs_item_description,batchInputs_price)

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