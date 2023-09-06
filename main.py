from flask import Flask,render_template,request
import pandas as pd
import pickle

app = Flask(__name__)
data=pd.read_csv("Cleaned_data.csv")

pipe=pickle.load(open("final.pkl",'rb'))

@app.route('/')
def index():
    locations=sorted(data['location'].unique())
    return render_template('index.html',locations=locations)


@app.route('/predict',methods=['POST'])
def predict():
    location=request.form.get('location')
    total_sqft=request.form.get('total_sqft')
    bath=request.form.get('bath')
    balcony=request.form.get('balcony')
    bedrooms=request.form.get('bedrooms')
    print(location,total_sqft,bath,balcony,bedrooms)
    input=pd.DataFrame([[location,total_sqft,bath,balcony,bedrooms]],columns=['location','total_sqft','bath','balcony','bedrooms'])
    prediction=pipe.predict(input)[0]

    return str(prediction)
if __name__=="__main__":
    app.run(debug=True,port=5001)
