from flask import Flask, render_template, request, redirect

app = Flask(__name__)

def predict_price(a,b,c,d,e,f):
    # TODO Move to model/inference later
    return 3000

@app.route("/")
def home():
    return render_template("price-prediction.html")


@app.route("/", methods=["POST"])
def result():
    if request.method == "POST":
        # Get data from html form
        req = request.form
        name = req['name']
        item_condition_id = req['item_condition_id']
        category_name = req['category_name']
        brand_name = req['brand_name']
        shipping = req['shipping']
        item_description = req['item_description']

        # Predict the price 
        price = predict_price(name,item_condition_id,category_name,brand_name,shipping,item_description)
        return render_template("price-prediction.html", price=price)

if __name__ == "__main__":
    app.run(debug=True)