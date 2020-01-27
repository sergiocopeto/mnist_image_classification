"""
This script implements a flask API for image classification using different models
"""
import os

from flask import Flask, render_template, request, jsonify

from mnist_image_classification.service import ModelManager

# Initialize model manager, so we can have multiple models running at the same time
model_manager = ModelManager()

app = Flask(__name__, template_folder='templates')
app.config['EXPLAIN_TEMPLATE_LOADING'] = True
# Where to store the uploaded files
app.config["IMAGE_UPLOADS"] = "static/img/uploads/"

# If image storing folder does not exist, create it
if not os.path.exists(app.config["IMAGE_UPLOADS"]):
    os.makedirs(app.config["IMAGE_UPLOADS"])


@app.route("/predict_hog_svm", methods=["POST"])
def predict_hog_svm():
    """
    Endpoint for url image classification using HOG_SVM model
    """
    image_url = request.form.get('image')
    result = model_manager.predict_url(image_url, 'hog_svm')
    response = {'model': "hog_svm",
               'image_url': image_url,
               'result': int(result)}
    return jsonify(response)


@app.route("/predict_cnn_4layer", methods=["POST"])
def predict_cnn_4layer():
    """
    Endpoint for url image classification using CNN_4Layer model
    """
    image_url = request.form.get('image')
    result = model_manager.predict_url(image_url, 'cnn_4layer')
    response = {'model': "cnn_4layer",
               'image_url': image_url,
               'result': int(result)}
    return jsonify(response)


@app.route("/predict_vgg19", methods=["POST"])
def predict_vgg19():
    """
    Endpoint for url image classification using VGG19 model
    """
    image_url = request.form.get('image')
    result = model_manager.predict_url(image_url, 'vgg19')
    response = {'model': "vgg19",
               'image_url': image_url,
               'result': int(result)}
    return jsonify(response)


@app.route("/", methods=["GET", "POST"])
def home():
    """
    This endpoint provides an interface for image uploading and model selection
    """
    print(app.config['APPLICATION_ROOT'])
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            model = request.form.get('models')
            image_file = os.path.join(app.config["IMAGE_UPLOADS"], image.filename)
            image.save(image_file)
            result = model_manager.predict_local(image_file, model)
            return "Predicted number: " + str(result)
    return render_template("home.html")
