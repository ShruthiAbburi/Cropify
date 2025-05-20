# from __future__ import print_function
# from flask import Flask, request, jsonify,redirect,render_template
# import pandas as pd
# import numpy as np
# import pickle
# import requests
# import warnings
# warnings.filterwarnings('ignore')
# from json import *
# from flask_cors import CORS, cross_origin
# import os 
# import joblib
# import CNN
# import torch
# from torchvision.io import read_image
# import torchvision.transforms as TF
# import torchvision.transforms.functional as MF

# from PIL import Image


# app = Flask(__name__)
# CORS(app)


# crop_model_path = 'models/NBClassifier.pkl'
# fertilizer_model_path = 'models/fert_DT.pkl'
# yield_model_path = 'models/xg_boost_tuned_model.pkl'


# rainfall_data = 'data2.csv'

# crop_recommendation_model = pickle.load(
#         open(crop_model_path, "rb"))

# fertilizer_recommendation_model = pickle.load(
#         open(fertilizer_model_path, "rb"))


# yield_model = joblib.load(yield_model_path)


# @app.route("/crop", methods=["POST"])
# def members1():
#     try:
#         N = int(request.json["N"])
#         P = int(request.json["P"])
#         K = int(request.json["K"])

#         ph = float(request.json["Ph"])
#         state = request.json["state"]
#         district = request.json["district"]
#         start_month = int(request.json["start_month"])
#         end_month = int(request.json["end_month"])
#     except:
#         return jsonify({"crop": "failed to get crop information", "data":request.json})

#     temperature = 20
#     humidity = 30
#     rainfall = 100

#     # getting the location using API 
#     x = requests.get(f"https://api.mapbox.com/geocoding/v5/mapbox.places/{district} {state}.json?access_token=pk.eyJ1Ijoic2FpZ29ydGk4MSIsImEiOiJja3ZqY2M5cmYydXd2MnZwZ2VoZzl1ejNkIn0.CupGYvpb_LNtDgp7b-rZJg")

#     coordinates = x.json()["features"][0]["center"]

#     # getting the humidity and temperature using API
#     y = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat={str(coordinates[1])}&lon={str(coordinates[0])}&appid=8d51fbf3b5ad7f3cc65ba0ea07220782")
#     humidity = y.json()["main"]["humidity"]
#     temperature = y.json()["main"]["temp"]

#     df = pd.read_csv(rainfall_data)
    
#     q = df.query(f'STATE_UT_NAME == "{state}" and DISTRICT == "{district}"')

#     total = 0
#     l = 0

#     if start_month <= end_month:
#         l = (end_month-start_month)+1

#         for i in range(start_month, end_month+1):
#             try:
#                 total+=int(q[i:i+1].value)
#             except:
#                 total-=1
#     elif start_month > end_month:
#         l = (end_month+12) - start_month + 1
        
#         for i in range(start_month, 13):
#             try:
#                 total+=int(q[i:i+1].value)
#             except:
#                 total-=1
        
#         for i in range(1, end_month+1):
#             try:
#                 total+=int(q[i:i+1].value)
#             except:
#                 total-=1


#     avg_rainfall = total/l

#     data = np.array([[N,P,K, temperature, humidity, ph, avg_rainfall]])

#     whole_prediction = crop_recommendation_model.predict(data)
#     prediction = whole_prediction[0]

#     return jsonify({"crop": prediction, "data":y.json()["main"], "l":l})


# @app.route("/fertilizer", methods=["POST"])
# def members2():
#     try:
#         N = int(request.json['N'])
#         P = int(request.json['P'])
#         K = int(request.json['K'])
#         # ph = float(request.json['Ph'])
#         state = request.json['state']
#         district = request.json['district']
#         moisture = float(request.json['moisture'])
#         soil_type = request.json['soil_type']
#         crop_type = request.json['crop_type']
#         start_month = int(request.json['start_month'])
#         end_month = int(request.json['end_month'])
#     except:
#         return jsonify({"crop": 'failed to get fertilizer information', "data": request.json})

#     temprature = 20
#     humidity = 30
#     rainfall = 100
    
#     x = requests.get(f"https://api.mapbox.com/geocoding/v5/mapbox.places/{district}{state}.json?access_token=pk.eyJ1Ijoic2FpZ29ydGk4MSIsImEiOiJja3ZqY2M5cmYydXd2MnZwZ2VoZzl1ejNkIn0.CupGYvpb_LNtDgp7b-rZJg")
#     coordinates =  x.json()['features'][0]['center']

#     y = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat={str(coordinates[1])}&lon={str(coordinates[0])}&appid=8d51fbf3b5ad7f3cc65ba0ea07220782")
#     humidity = y.json()['main']['humidity']
#     temprature = y.json()['main']['temp']

#     df=pd.read_csv("./data2.csv")
#     q = df.query('STATE_UT_NAME=="ANDAMAN And NICOBAR ISLANDS" and DISTRICT == "NICOBAR"', inplace = False)

#     total = 0
#     l = 0

#     if start_month <= end_month: 
#         l=(end_month-start_month)+1

#         for i in range(start_month, end_month+1):
#             try:
#                 total+=int(q[i:i+1].value)
#             except:
#                 total-=1
            
#     elif start_month > end_month:
#         l = (end_month+12) - start_month + 1
        
#         for i in range(start_month, 13):
#             try:
#                 total+=int(q[i:i+1].value)
#             except:
#                 total-=1
        
#         for i in range(1, end_month+1):
#             try:
#                 total+=int(q[i:i+1].value)
#             except:
#                 total-=1

#     avg_rainfall = total/l


#     data = np.array([[avg_rainfall, humidity, moisture, soil_type, crop_type, N, K, P]])


#     whole_prediction = fertilizer_recommendation_model.predict(data)
#     prediction = whole_prediction[0]


#     fertname = {"10-26-26": "Suggested Brand : Gromor 10-26-26", "14-35-14": "Suggested Brand : Sansar Green 14-35-14", "17-17-17": "Suggested Brand : Mangala 17-17-17", "20-20": "Suggested Brand : Ravk Kvar 20-20", "28-28": "Suggested Brand : Coromondal Gromor 28-28", "DAP": "Suggested Brand : DAP", "Urea": "Suggested Brand : YaraVera"}

#     fertilizer = fertname.get(str(prediction))
#     return jsonify({"crop": str(prediction) , "data": fertname, 'Fertilizer': fertilizer})



# @app.route("/yield", methods=["POST"])
# def members3():
#     try:
#         # District = 15
#         CultLand = float(request.json['CultLand'])
#         CropCultLand = float(request.json['CropCultLand']) 

#         # CropTillageDate = float(request.json["CropTillageDate"])
        
#         SeedlingsPerPit = float(request.json["SeedlingsPerPit"])

#         TransplantingIrrigationHours = float(request.json["TransplantingIrrigationHours"])

#         # TransplantingIrrigationSource = float(request.json["TransplantingIrrigationSource"])

#         # TransplantingIrrigationPowerSource = float(request.json["TransplantingIrrigationPowerSource"])

#         # StandingWater = float(request.json["StandingWater"])

#         BasalDAP = float(request.json["BasalDAP"])

#         # BasalUrea = float(request.json["BasalUrea"])

#         # FirstTopDressFert = float(request.json["FirstTopDressFert"])

#         # _1tdUrea = float(request.json["_1tdUrea"])

#         # MineralFertAppMethod = float(request.json["MineralFertAppMethod"])

#         # Harv_method = float(request.json["Harv_method"])

#         # Harv_date = float(request.json["Harv_date"])

#         # Threshing_date = float(request.json["Threshing_date"])

#         Acre = float(request.json["Acre"])

#         # Block = float(request.json["Block"])

#         # LandPreparationMethod = float(request.json["LandPreparationMethod"])

#         # CropTillageDepth = float(request.json["CropTillageDepth"])

#         # CropEstMethod = float(request.json["CropEstMethod"])

#         # NursDetFactor = float(request.json["NursDetFactor"])

#         # TransDetFactor = float(request.json["TransDetFactor"])

#         # TransIrriCost = float(request.json["TransIrriCost"])

#         # OrgFertilizers = float(request.json["OrgFertilizers"])

#         # Ganaura = float(request.json["Ganaura"])

#         # CropOrgFYM = float(request.json["CropOrgFYM"])

#         # PCropSolidOrgFertAppMethod = float(request.json["PCropSolidOrgFertAppMethod"])

#         # NoFertilizerAppln = float(request.json["NoFertilizerAppln"])

#         # CropbasalFerts = float(request.json["CropbasalFerts"])
        
#         features = [5, CultLand, CropCultLand, 8, SeedlingsPerPit,
#                             TransplantingIrrigationHours, 5,
#                             15, 0.5, BasalDAP,
#                             1, 2, 3, 4,
#                             5, 6.0, 4.0, Acre, 7,
#                             4, 5, 3,
#                             5, 4, 2, 0.5,
#                             4, 2, 4,
#                             6, 10]
        
#         n = []
#         for i in features:
#             n.append(float(i))
#         data3 = np.array([n])
#         whole_prediction3 = yield_model.predict(data3)
#         prediction3 = whole_prediction3[0]
#         print(prediction3)
        
#         return jsonify({"yield":prediction3})

#     except Exception as e:
#         print(e)
#         return jsonify({"error": "Failed to process the request."})


# # leaf new started
# # leaf new started
# # leaf new started

# disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
# supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')

# model = CNN.CNN(39)    
# model.load_state_dict(torch.load("C:\\Users\\Atheek\\Documents\\crop\\my-app\\server\\models\\plant_disease_model_1_latest.pt"))
# model.eval()

# def prediction(image_path):
#     image = Image.open(image_path)
#     image = image.resize((224, 224))
#     input_data = MF.to_tensor(image)
#     input_data = input_data.view((-1, 3, 224, 224))
#     output = model(input_data)
#     output = output.detach().numpy()
#     index = np.argmax(output)
#     return index

# @app.route('/home_leaf')
# def leaf_home():
#     return render_template('home_leaf.html')

# @app.route('/index_leaf')
# def index_leaf():
#     return render_template('index_leaf.html')

# @app.route('/submit', methods=['GET', 'POST'])
# def submit():
#     if request.method == 'POST':
#         image = request.files['image']
#         filename = image.filename
#         file_path = os.path.join('static/uploads', filename)
#         image.save(file_path)
#         print(file_path)
#         pred = prediction(file_path)
#         title = disease_info['disease_name'][pred]
#         description =disease_info['description'][pred]
#         prevent = disease_info['Possible Steps'][pred]
#         image_url = disease_info['image_url'][pred]
#         supplement_name = supplement_info['supplement name'][pred]
#         supplement_image_url = supplement_info['supplement image'][pred]
#         supplement_buy_link = supplement_info['buy link'][pred]
#         return render_template('submit.html' , title = title , desc = description , prevent = prevent , 
#                                image_url = image_url , pred = pred ,sname = supplement_name , simage = supplement_image_url , buy_link = supplement_buy_link)

# @app.route('/market', methods=['GET', 'POST'])
# def market():
#     return render_template('market.html', supplement_image = list(supplement_info['supplement image']),
#                            supplement_name = list(supplement_info['supplement name']), disease = list(disease_info['disease_name']), buy = list(supplement_info['buy link']))


# @app.route("/leaf", methods=["POST"])
# @cross_origin()
# def members4():
#     print("hello from leaf")
#     try:
#         # Get the file from the form data
#         file = request.files.get('file')
        
#         if not file:
#             return jsonify({'error': 'No file part'})
        
#         # Check if the file has a valid filename
#         if file.filename == '':
#             return jsonify({'error': 'No selected file'})
        
#         # Check if the file has a valid extension
#         allowed_extensions = {'jpg'}
#         if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
#             return jsonify({'error': 'Only JPG files are allowed'})
        
#         # Save the file
#         file.save('research/leaf.jpg')
#         print("File saved in research folder")

               
#         # Load and preprocess the image
#         new_img = Image.open('leaf_images/preprocessed_image.jpg').convert('RGB').resize((224, 224))
#         img = TF.ToTensor()(new_img)
#         img = img.unsqueeze(0)
#         img = img / 255.0

#         # Save the preprocessed image
#         # img_numpy = img.squeeze(0).permute(1, 2, 0).numpy()
#         # img = Image.fromarray((img_numpy * 255).astype('uint8'))

#         # Load the model and make a prediction
#         model = CNN.CNN(39)
#         model.load_state_dict(torch.load('plant_disease_model_1_latest.pt', map_location=torch.device('cpu')))
#         prediction = model(img).detach().numpy()

#         predicted_class_idx = np.argmax(prediction[0])

#         class_labels = [
#             'Apple___Apple_scab',
#             'Apple___Black_rot',
#             'Apple___Cedar_apple_rust',
#             'Apple___healthy',
#             'Background_without_leaves',
#             'Blueberry___healthy',
#             'Cherry___Powdery_mildew',
#             'Cherry___healthy',
#             'Corn___Cercospora_leaf_spot Gray_leaf_spot',
#             'Corn___Common_rust',
#             'Corn___Northern_Leaf_Blight',
#             'Corn___healthy',
#             'Grape___Black_rot',
#             'Grape___Esca_(Black_Measles)',
#             'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
#             'Grape___healthy',
#             'Orange___Haunglongbing_(Citrus_greening)',
#             'Peach___Bacterial_spot',
#             'Peach___healthy',
#             'Pepper,_bell___Bacterial_spot',
#             'Pepper,_bell___healthy',
#             'Potato___Early_blight',
#             'Potato___Late_blight',
#             'Potato___healthy',
#             'Raspberry___healthy',
#             'Soybean___healthy',
#             'Squash___Powdery_mildew',
#             'Strawberry___Leaf_scorch',
#             'Strawberry___healthy',
#             'Tomato___Bacterial_spot',
#             'Tomato___Early_blight',
#             'Tomato___Late_blight',
#             'Tomato___Leaf_Mold',
#             'Tomato___Septoria_leaf_spot',
#             'Tomato___Spider_mites Two-spotted_spider_mite',
#             'Tomato___Target_Spot',
#             'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
#             'Tomato___Tomato_mosaic_virus',
#             'Tomato___healthy'
#         ]

#         # Get the predicted label
#         predicted_label = class_labels[predicted_class_idx]

#         print(f"Predicted Label: {predicted_label}")

#         return jsonify({"leaf status": predicted_label})
    
#     except Exception as e:
#         print(e)
#         return jsonify({"error": "Failed to process the request."})


# if __name__ == "__main__":
#     app.run(debug=True)

from __future__ import print_function
from flask import Flask, request, jsonify,redirect,render_template
import pandas as pd
import numpy as np
import pickle
import requests
import warnings
warnings.filterwarnings('ignore')
from json import *
from flask_cors import CORS, cross_origin
import os 
import joblib
import CNN
import torch
from torchvision.io import read_image
import torchvision.transforms as TF
import torchvision.transforms.functional as MF
import gdown

from PIL import Image


app = Flask(__name__)
CORS(app)


crop_model_path = 'models/NBClassifier.pkl'
fertilizer_model_path = 'models/fert_DT.pkl'
# yield_model_path = 'models/xg_boost_tuned_model.pkl'


rainfall_data = 'data2.csv'

crop_recommendation_model = pickle.load(
        open(crop_model_path, "rb"))

fertilizer_recommendation_model = pickle.load(
        open(fertilizer_model_path, "rb"))


# yield_model = joblib.load(yield_model_path)


@app.route("/crop", methods=["POST"])
def members1():
    try:
        N = int(request.json["N"])
        P = int(request.json["P"])
        K = int(request.json["K"])

        ph = float(request.json["Ph"])
        state = request.json["state"]
        district = request.json["district"]
        start_month = int(request.json["start_month"])
        end_month = int(request.json["end_month"])
    except:
        return jsonify({"crop": "failed to get crop information", "data":request.json})

    temperature = 20
    humidity = 30
    rainfall = 100

    # getting the location using API 
    x = requests.get(f"https://api.mapbox.com/geocoding/v5/mapbox.places/{district} {state}.json?access_token=pk.eyJ1Ijoic2FpZ29ydGk4MSIsImEiOiJja3ZqY2M5cmYydXd2MnZwZ2VoZzl1ejNkIn0.CupGYvpb_LNtDgp7b-rZJg")

    coordinates = x.json()["features"][0]["center"]

    # getting the humidity and temperature using API
    y = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat={str(coordinates[1])}&lon={str(coordinates[0])}&appid=8d51fbf3b5ad7f3cc65ba0ea07220782")
    humidity = y.json()["main"]["humidity"]
    temperature = y.json()["main"]["temp"]

    df = pd.read_csv(rainfall_data)
    
    q = df.query(f'STATE_UT_NAME == "{state}" and DISTRICT == "{district}"')

    total = 0
    l = 0

    if start_month <= end_month:
        l = (end_month-start_month)+1

        for i in range(start_month, end_month+1):
            try:
                total+=int(q[i:i+1].value)
            except:
                total-=1
    elif start_month > end_month:
        l = (end_month+12) - start_month + 1
        
        for i in range(start_month, 13):
            try:
                total+=int(q[i:i+1].value)
            except:
                total-=1
        
        for i in range(1, end_month+1):
            try:
                total+=int(q[i:i+1].value)
            except:
                total-=1


    avg_rainfall = total/l

    data = np.array([[N,P,K, temperature, humidity, ph, avg_rainfall]])

    whole_prediction = crop_recommendation_model.predict(data)
    prediction = whole_prediction[0]

    return jsonify({"crop": prediction, "data":y.json()["main"], "l":l})


@app.route("/fertilizer", methods=["POST"])
def members2():
    try:
        N = int(request.json['N'])
        P = int(request.json['P'])
        K = int(request.json['K'])
        # ph = float(request.json['Ph'])
        state = request.json['state']
        district = request.json['district']
        moisture = float(request.json['moisture'])
        soil_type = request.json['soil_type']
        crop_type = request.json['crop_type']
        start_month = int(request.json['start_month'])
        end_month = int(request.json['end_month'])
    except:
        return jsonify({"crop": 'failed to get fertilizer information', "data": request.json})

    temprature = 20
    humidity = 30
    rainfall = 100
    
    x = requests.get(f"https://api.mapbox.com/geocoding/v5/mapbox.places/{district}{state}.json?access_token=pk.eyJ1Ijoic2FpZ29ydGk4MSIsImEiOiJja3ZqY2M5cmYydXd2MnZwZ2VoZzl1ejNkIn0.CupGYvpb_LNtDgp7b-rZJg")
    coordinates =  x.json()['features'][0]['center']

    y = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat={str(coordinates[1])}&lon={str(coordinates[0])}&appid=8d51fbf3b5ad7f3cc65ba0ea07220782")
    humidity = y.json()['main']['humidity']
    temprature = y.json()['main']['temp']

    df=pd.read_csv("./data2.csv")
    q = df.query('STATE_UT_NAME=="ANDAMAN And NICOBAR ISLANDS" and DISTRICT == "NICOBAR"', inplace = False)

    total = 0
    l = 0

    if start_month <= end_month: 
        l=(end_month-start_month)+1

        for i in range(start_month, end_month+1):
            try:
                total+=int(q[i:i+1].value)
            except:
                total-=1
            
    elif start_month > end_month:
        l = (end_month+12) - start_month + 1
        
        for i in range(start_month, 13):
            try:
                total+=int(q[i:i+1].value)
            except:
                total-=1
        
        for i in range(1, end_month+1):
            try:
                total+=int(q[i:i+1].value)
            except:
                total-=1

    avg_rainfall = total/l


    data = np.array([[avg_rainfall, humidity, moisture, soil_type, crop_type, N, K, P]])


    whole_prediction = fertilizer_recommendation_model.predict(data)
    prediction = whole_prediction[0]


    fertname = {"10-26-26": "Suggested Brand : Gromor 10-26-26", "14-35-14": "Suggested Brand : Sansar Green 14-35-14", "17-17-17": "Suggested Brand : Mangala 17-17-17", "20-20": "Suggested Brand : Ravk Kvar 20-20", "28-28": "Suggested Brand : Coromondal Gromor 28-28", "DAP": "Suggested Brand : DAP", "Urea": "Suggested Brand : YaraVera"}

    fertilizer = fertname.get(str(prediction))
    return jsonify({"crop": str(prediction) , "data": fertname, 'Fertilizer': fertilizer})



# @app.route("/yield", methods=["POST"])
# def members3():
#     try:
#         # District = 15
#         CultLand = float(request.json['CultLand'])
#         CropCultLand = float(request.json['CropCultLand']) 

#         # CropTillageDate = float(request.json["CropTillageDate"])
        
#         SeedlingsPerPit = float(request.json["SeedlingsPerPit"])

#         TransplantingIrrigationHours = float(request.json["TransplantingIrrigationHours"])

#         # TransplantingIrrigationSource = float(request.json["TransplantingIrrigationSource"])

#         # TransplantingIrrigationPowerSource = float(request.json["TransplantingIrrigationPowerSource"])

#         # StandingWater = float(request.json["StandingWater"])

#         BasalDAP = float(request.json["BasalDAP"])

#         # BasalUrea = float(request.json["BasalUrea"])

#         # FirstTopDressFert = float(request.json["FirstTopDressFert"])

#         # _1tdUrea = float(request.json["_1tdUrea"])

#         # MineralFertAppMethod = float(request.json["MineralFertAppMethod"])

#         # Harv_method = float(request.json["Harv_method"])

#         # Harv_date = float(request.json["Harv_date"])

#         # Threshing_date = float(request.json["Threshing_date"])

#         Acre = float(request.json["Acre"])

#         # Block = float(request.json["Block"])

#         # LandPreparationMethod = float(request.json["LandPreparationMethod"])

#         # CropTillageDepth = float(request.json["CropTillageDepth"])

#         # CropEstMethod = float(request.json["CropEstMethod"])

#         # NursDetFactor = float(request.json["NursDetFactor"])

#         # TransDetFactor = float(request.json["TransDetFactor"])

#         # TransIrriCost = float(request.json["TransIrriCost"])

#         # OrgFertilizers = float(request.json["OrgFertilizers"])

#         # Ganaura = float(request.json["Ganaura"])

#         # CropOrgFYM = float(request.json["CropOrgFYM"])

#         # PCropSolidOrgFertAppMethod = float(request.json["PCropSolidOrgFertAppMethod"])

#         # NoFertilizerAppln = float(request.json["NoFertilizerAppln"])

#         # CropbasalFerts = float(request.json["CropbasalFerts"])
        
#         features = [5, CultLand, CropCultLand, 8, SeedlingsPerPit,
#                             TransplantingIrrigationHours, 5,
#                             15, 0.5, BasalDAP,
#                             1, 2, 3, 4,
#                             5, 6.0, 4.0, Acre, 7,
#                             4, 5, 3,
#                             5, 4, 2, 0.5,
#                             4, 2, 4,
#                             6, 10]
        
#         n = []
#         for i in features:
#             n.append(float(i))
#         data3 = np.array([n])
#         whole_prediction3 = yield_model.predict(data3)
#         prediction3 = whole_prediction3[0]
#         print(prediction3)
        
#         return jsonify({"yield":prediction3})

#     except Exception as e:
#         print(e)
#         return jsonify({"error": "Failed to process the request."})


# leaf new started
# leaf new started
# leaf new started

disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')

model = CNN.CNN(39)
# model_pt = gdown.download("https://drive.google.com/file/d/1CeKAroGvJLDFWXR8f_A7QgFXWZ7qbHD8/view?usp=sharing","models")

model.load_state_dict(torch.load("models\\plant_disease_model_1_latest.pt"))
model.eval()

def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = MF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index

@app.route('/home_leaf')
def leaf_home():
    return render_template('home_leaf.html')

@app.route('/index_leaf')
def index_leaf():
    return render_template('index_leaf.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
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


@app.route("/leaf", methods=["POST"])
@cross_origin()
def members4():
    print("hello from leaf")
    try:
        # Get the file from the form data
        file = request.files.get('file')
        
        if not file:
            return jsonify({'error': 'No file part'})
        
        # Check if the file has a valid filename
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        # Check if the file has a valid extension
        allowed_extensions = {'jpg'}
        if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return jsonify({'error': 'Only JPG files are allowed'})
        
        # Save the file
        file.save('research/leaf.jpg')
        print("File saved in research folder")

               
        # Load and preprocess the image
        new_img = Image.open('leaf_images/preprocessed_image.jpg').convert('RGB').resize((224, 224))
        img = TF.ToTensor()(new_img)
        img = img.unsqueeze(0)
        img = img / 255.0

        # Save the preprocessed image
        # img_numpy = img.squeeze(0).permute(1, 2, 0).numpy()
        # img = Image.fromarray((img_numpy * 255).astype('uint8'))

        # Load the model and make a prediction
        model = CNN.CNN(39)
        model.load_state_dict(torch.load('plant_disease_model_1_latest.pt', map_location=torch.device('cpu')))
        prediction = model(img).detach().numpy()

        predicted_class_idx = np.argmax(prediction[0])

        class_labels = [
            'Apple___Apple_scab',
            'Apple___Black_rot',
            'Apple___Cedar_apple_rust',
            'Apple___healthy',
            'Background_without_leaves',
            'Blueberry___healthy',
            'Cherry___Powdery_mildew',
            'Cherry___healthy',
            'Corn___Cercospora_leaf_spot Gray_leaf_spot',
            'Corn___Common_rust',
            'Corn___Northern_Leaf_Blight',
            'Corn___healthy',
            'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)',
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)',
            'Peach___Bacterial_spot',
            'Peach___healthy',
            'Pepper,_bell___Bacterial_spot',
            'Pepper,_bell___healthy',
            'Potato___Early_blight',
            'Potato___Late_blight',
            'Potato___healthy',
            'Raspberry___healthy',
            'Soybean___healthy',
            'Squash___Powdery_mildew',
            'Strawberry___Leaf_scorch',
            'Strawberry___healthy',
            'Tomato___Bacterial_spot',
            'Tomato___Early_blight',
            'Tomato___Late_blight',
            'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]

        # Get the predicted label
        predicted_label = class_labels[predicted_class_idx]

        print(f"Predicted Label: {predicted_label}")

        return jsonify({"leaf status": predicted_label})
    
    except Exception as e:
        print(e)
        return jsonify({"error": "Failed to process the request."})


if __name__ == "__main__":
    app.run(debug=True)
