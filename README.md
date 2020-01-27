# MNIST Image Classification

This repository performs some experiments using different image classification models using the MNIST dataset.
It also provides a simple Flask API for classification of images provided by URL's or through it's uploading interface.

## Setup
Clone this repository, navigate into it through the command line and run the following command:

```
pip install -r requirements.txt
```

This command should install all the needed packages

## Usage
### Model Training
In order to train the different models, a configuration file similar to this is provided:

````
{
  "train_images_path": "data/train-images.idx3-ubyte",
  "train_labels_path": "data/train-labels.idx1-ubyte",
  "number_of_samples": 500,
  "augment_images": true,
  "number_of_augmentations": 5,
  "train_hog_svm": true,
  "train_cnn_1layer": true,
  "train_cnn_3layer": true,
  "train_cnn_4layer": true,
  "train_vgg19": true,
  "output_models_folder": "output_models/",
  "output_metrics_report": "results_with_augmentation.json"
}
````

* train_images_path: Path to the file containing the MNIST training images
* train_labels_path: Path to the file containing the MNIST training labels
* number_of_samples: Number of samples to collect
* augment_images: Whether or not to perform dataset augmentation. This process includes the following techniques:
    * Shearing
    * Shifting
    * Zoom
    * Rotation
* number_of_augmentations: Number of times the augmentation is done for each image
* train_hog_svm: True if one wants to train a model using Histogram of Oriented Gradients
* train_cnn_1layer: True if one wants to train a neural network using a single convolutional layer
* train_cnn_3layer: True if one wants to train a neural network using 3 convolutional layers
* train_cnn_4layer: True if one wants to train a neural network using 4 convolutional layers
* train_vgg19: True if one wants to train a neural network based on a pre-trained VGG19 model
* output_models_folder: Path to folder where to store the trained models and their performance metrics
* output_metrics_report: Path to a JSON file that will contain the relevant metrics for all the experiments ran with this configuration file

Regarding the two output fields, the models folder output will be used to store the weights for the trained model in .h5 format, as well as
a json file with the same name, which will contain the full classification report for the stored model.
The metrics report output will contain the original configuration for the target experiment, and all the relevant metrics
for all the trained models. Please take a look at the "results_with_augmentation.json" file, provided as an example.

As such, one can run an experiment using the following command:

````
python model_training.py -i config.json
````

### Service

The service can be started with the following command:

````
python run_service.py -i config_service.json
````

Where config_service.json contains the models to be loaded. As an example:

````
{
  "hog_smv_path": "output_models/HogSVM_20200127_105836.h5",
  "cnn_4layer_path": "output_models/Cnn4Layer_20200127_110622.h5",
  "vgg_19_path": "output_models/VGG_19_20200127_110721.h5"
}
````

There are some useful API endpoints:

* /predict_hog_svm (Method: POST): Perform classification using a HOG_SVM model
* /predict_cnn_4layer (Method: POST): Perform classification using a CNN_4Layer model
* /predict_vgg19 (Method: POST): : Perform classification using a VGG19 model

For the body of the POST message, the user should provide the following data:

````
{
    "image": "http://bradleymitchell.me/wp-content/uploads/2014/06/decompressed.jpg"
}
````

The result of this call should look like this:

````
{
    "image_url": "http://bradleymitchell.me/wp-content/uploads/2014/06/decompressed.jpg",
    "model": "<The model selected in the endpoint>",
    "result": 5
}
````

#### Root endpoint
The Root endpoint can be opened in the browser. It is composed by a simple interface where the user can upload a local image
and select a model for classification.