import json
from datetime import datetime

import numpy as np
from keras import layers
from keras import models
from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.image import img_to_array, array_to_img
from keras.utils import to_categorical
from sklearn.metrics import classification_report

from mnist_image_classification.models.cnn_base import CnnBase


class VGG_19(CnnBase):
    """
    This class implements an image classification neural network based on a pre-trained VGG19 model
    """
    def __init__(self):
        """
        Class initialization.
        """
        super().__init__()
        self.build_model()

    def build_model(self):
        """
        Builds an image classification neural network based on a pre-trained VGG19 model
        :return:
        """
        self.vgg19 = VGG19(weights='imagenet', include_top=False, input_shape = (32, 32, 3), classes = 10)

        # Add Dense and Dropout layers on top of VGG19 pre-trained
        self.model = models.Sequential()
        self.model.add(layers.Dense(512, activation='relu', input_dim=1 * 1 * 512))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(10, activation="softmax"))

        # Compile the model
        self.model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    def compute_features(self, _images):
        images = np.dstack([_images] * 3)
        images = images.reshape(-1, 28, 28, 3)
        images = np.asarray([img_to_array(array_to_img(im, scale=False).resize((32, 32))) for im in images])
        X_train = preprocess_input(images)

        # Extracting features
        train_features = self.vgg19.predict(np.array(X_train), batch_size=256, verbose=1)
        # Flatten extracted features
        features = np.reshape(train_features, (len(images), 1 * 1 * 512))
        return features

    def train(self, train_images, train_labels, _epochs=10):
        """
        This method implement the training routine for a CNN based neural network.
        :param images: Array of images used as training samples
        :param labels: Array of corresponding labels
        :param _epochs: Number of training epochs
        """
        labels = to_categorical(train_labels)
        train_features = self.compute_features(train_images)
        self.model.fit(train_features, labels, epochs=_epochs)

    def predict(self, _images):
        """
        Method for image classification of a given set of images
        :param _images: Array containing the set of images to be classifier
        :return: Array containing the probabilities for each class, for each input image
        """
        features = self.compute_features(_images)
        return self.model.predict(features)

    def eval(self, images, labels):
        """
        Performs evaluation of a trained neural network
        :param images: Array containing the test images
        :param labels: Array containing labels corresponding to the test images
        :return: Classification report for experiment management
        """
        features = self.compute_features(images)
        predicted_classes = self.model.predict(features)
        predicted_classes = [np.argmax(prediction) for prediction in predicted_classes]
        print(predicted_classes)

        target_names = ["Class {}".format(i) for i in range(10)]
        class_report = classification_report(labels, predicted_classes, target_names=target_names, output_dict=True)
        print(class_report)
        return class_report

    def train_and_eval(self, _train_images, _train_labels, _test_images, _test_labels, _epochs=10, save_path = ''):
        """
        This method implements full training and evaluation and saving routine for CNN models
        :param _train_images: Array containing the training images
        :param _train_labels: Array containing the training labels
        :param _test_images: Array containing the testing images
        :param _test_labels: Array containing the testing labels
        :param _epochs: Number of training epochs
        :param save_path: A folder where to store the trained model
        :return: Dictionary containing performance metrics for the trained model
        """
        self.train(_train_images, _train_labels, _epochs)
        report = self.eval(_test_images, _test_labels)
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        weights_file = save_path + self.name() + '_' + now + '.h5'
        report_file = save_path + self.name() + '_' + now + '.json'
        self.save(weights_file)
        # build report
        output_dict = {'model_name': self.name(),
                       'weights_file': weights_file,
                       'classification_report': report}
        with open(report_file, 'w') as f:
            json.dump(output_dict, f)
        return output_dict
