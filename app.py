from flask import Flask, render_template, request, redirect
from inference import Inference, DataProcessor
import pickle
import numpy as np
from contextlib import contextmanager
import time
app = Flask(__name__)

def predict_price(a,b,c,d,e,f):
    # TODO Move to model/inference.py later
    return 3000

@app.route("/")
def home():
    return render_template("price-prediction.html")

def process_incoming_data(data):
    return data

def preload():
    global predictor, data_processor
    predictor = Inference()
    data_processor = DataProcessor('sent2vec/wiki_unigrams.bin',"preprocess/category_encoding.pickle", "preprocess/brand_encoding.pickle")
    
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print("{} done in {}s".format(name, time.time() - t0))

@app.route("/", methods=["POST"])
def result():
    global predictor, data_processor
    if request.method == "POST":
        # Get data from html form
        req = request.form
        name = req['name']
        item_condition_id = req['item_condition_id']
        category_name = req['category_name']
        brand_name = req['brand_name']
        shipping = req['shipping']
        item_description = req['item_description']
        data = (name,item_condition_id,category_name,brand_name,shipping,item_description)
        
        processed_data = data_processor.process(data)
        # Predict the price 
        price = predictor.make_prediction(processed_data) 
        return render_template("price-prediction.html", price=price)
    else:
        return render_template("price-prediction.html")

if __name__ == "__main__":
    with timer("Preload Model"):
        preload()
    app.run(debug=True)