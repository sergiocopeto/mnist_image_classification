import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from sklearn.metrics import classification_report


class CnnBase():
    """
    This class serves as a base class for convolutional neural network building.
    It includes all the relevant methods for training and evaluating neural networks, as well as method for prediction,
    model loading (including the training and evaluation results) and saving.
    """
    def __init__(self):
        """
        Class initialization.
        """
        self.model = None
        self.history = None

    def name(self):
        """
        Returns class name. This is useful to track all the different experiments and results
        :return: Class name
        """
        return self.__class__.__name__

    def build_model(self):
        """
        Method for building the neural network architecture. To be implemented by the child classes
        """
        pass

    def train(self, images, labels, _epochs=10):
        """
        This method implement the training routine for a CNN based neural network.
        :param images: Array of images used as training samples
        :param labels: Array of corresponding labels
        :param _epochs: Number of training epochs
        """
        labels = to_categorical(labels)
        self.history = self.model.fit(x=images, y=labels, epochs=_epochs)

    def predict(self, _images):
        """
        Method for image classification of a given set of images
        :param _images: Array containing the set of images to be classifier
        :return: Array containing the probabilities for each class, for each input image
        """
        # Convert all the images to the correct neural network input format
        images = np.array([x.reshape(x.shape[0], x.shape[1], 1) for x in _images])
        return self.model.predict(images, verbose=3)

    def eval(self, images, labels):
        """
        Performs evaluation of a trained neural network
        :param images: Array containing the test images
        :param labels: Array containing labels corresponding to the test images
        :return: Classification report for experiment management
        """
        predicted_classes = self.model.predict(images)
        predicted_classes = [np.argmax(x) for x in predicted_classes]
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
        # Prepare all images and labels for training and testing
        train_images = np.array([x.reshape(x.shape[0], x.shape[1], 1) for x in _train_images])
        test_images = np.array([x.reshape(x.shape[0], x.shape[1], 1) for x in _test_images])
        train_labels = _train_labels
        test_labels = _test_labels
        # Trains the model
        self.train(train_images, train_labels, _epochs)
        # Performs model evaluation
        report = self.eval(test_images, test_labels)
        # Stores a snapshot of the model, as well as it's statistics
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

    def show_history(self):
        """
        Small method for plotting training history
        """
        if self.history is not None:
            plt.plot(self.history.history['accuracy'])
            plt.legend(['accuracy'])
            plt.title('Accuracy')
            plt.xlabel('epoch')
            plt.show()

    def save(self, path):
        """
        Saves the model weights
        :param path: Path to the file where to save the model weights
        """
        self.model.save_weights(path)

    def load(self, path):
        """
        Loads a trained model weights
        :param path: Path to the weights file
        """
        self.model.load_weights(path)
