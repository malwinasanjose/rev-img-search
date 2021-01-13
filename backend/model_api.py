from flask import Flask, jsonify, request
from joblib import load 
import numpy as np

app = Flask(__name__)

fpath_model = "./NearestNeighbors.joblib"  
CLF = load(fpath_model)

def get_nearest_ids(image_features):
    """
    use scikit-learn classifier to return the closest results
    """
    distances, indices = CLF.kneighbors(image_features)
    # .labels is not a standard attribute of NearestNeighbours. this was added during sklearn model training training for ease of future use
    object_ids = CLF.labels[indices.tolist()[0]]
    return object_ids

@app.route('/')
def home():
    return """
    <h1>Flask + Scikit-learn demo app</h1>
    <img width="200px" src="https://propulsion.academy/_nuxt/img/5a20d82.png"/>
    """

@app.route('/predict', methods=['POST'])
def predict():

    # access image_features values from the request 
    req_json = request.json
    image_features = req_json['image_features']
    # format image_features for model 
    image_features = np.array(image_features).reshape(1,-1)
    #get the nearest neighbours
    id_results = get_nearest_ids(image_features)
    id_results = id_results.tolist()
    # return json results
    return jsonify({"result": id_results})

if __name__ == '__main__':

    app.run(debug=True)