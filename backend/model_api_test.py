# informal, manual integration test for the model api
import requests
import joblib

url = "http://127.0.0.1:5000/predict"
image_test_set = joblib.load("test_set.joblib")
image_features = image_test_set[0].reshape(1,-1)
image_features = image_features.tolist()
response = requests.post(url, json={"image_features": image_features})


import requests
import joblib

url = "http://127.0.0.1:5000/predict"
image_test_set = joblib.load("test_set.joblib")
image_features = image_test_set[0].reshape(1,-1)
image_features = image_features.tolist()
response = requests.post(url, json={"image_features": image_features})

print("url returned response {}".format(response.status_code))
print("response content: ", response.json())
