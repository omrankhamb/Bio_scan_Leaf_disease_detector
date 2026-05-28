import os
from flask import Flask, redirect, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

disease_info = pd.read_csv(os.path.join(BASE_DIR, 'disease_info.csv'), encoding='cp1252')
supplement_info = pd.read_csv(os.path.join(BASE_DIR, 'supplement_info.csv'), encoding='cp1252')

model = CNN.CNN(39)    
model.load_state_dict(torch.load(os.path.join(BASE_DIR, "plant_disease_model_1_latest.pt")))
model.eval()

def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index


app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/apple')
def apple_page():
    return render_template('apple.html')

@app.route('/bluecherry')
def bluecherry_page():
    return render_template('bluecherry.html')

@app.route('/cherry')
def cherry_page():
    return render_template('cherry.html')

@app.route('/corn')
def corn_page():
    return render_template('corn.html')

@app.route('/grape')
def grape_page():
    return render_template('grape.html')

@app.route('/orange')
def orange_page():
    return render_template('orange.html')

@app.route('/peach')
def peach_page():
    return render_template('peach.html')

@app.route('/pepper')
def pepper_page():
    return render_template('pepper.html')

@app.route('/potato')
def potato_page():
    return render_template('potato.html')

@app.route('/raspberry')
def raspberry_page():
    return render_template('raspberry.html')

@app.route('/soyabean')
def soyabean_page():
    return render_template('soyabean.html')

@app.route('/squash')
def squash_page():
    return render_template('squash.html')

@app.route('/strawberry')
def strawberry_page():
    return render_template('strawberry.html')

@app.route('/tomato')
def tomato_page():
    return render_template('tomato.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join(BASE_DIR, 'static/uploads', filename)
        image.save(file_path)
        print(file_path)
        pred = prediction(file_path)
        title = disease_info['disease_name'][pred]
        description =disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        return render_template('submit.html' , title = title , desc = description , prevent = prevent , 
                               image_url = image_url , pred = pred ,sname = supplement_name , simage = supplement_image_url , buy_link = supplement_buy_link)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image = list(supplement_info['supplement image']),
                           supplement_name = list(supplement_info['supplement name']), disease = list(disease_info['disease_name']), buy = list(supplement_info['buy link']))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

