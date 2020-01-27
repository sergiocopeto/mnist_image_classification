import pickle
import json
import numpy as np
from datetime import datetime
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score

class HogSVM():
    """
    This class implements an image classification model based on an Histogram of Oriented Gradients
    and a Support Vector Machine
    """
    def __init__(self):
        """
        Class initialization
        """
        self.clf = None

    def name(self):
        """
        Returns class name. This is useful to track all the different experiments and results
        :return: Class name
        """
        return self.__class__.__name__

    def train(self, train_images, train_labels):
        """
        This method implement the training routine for a CNN based neural network.
        :param images: Array of images used as training samples
        :param labels: Array of corresponding labels
        """
        hog_features = []
        # Compute HOG features on the training data
        for image in train_images:
            fd, _ = hog(image, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(4, 4),
                                block_norm='L2', visualize=True)
            hog_features.append(fd)

        # Train a SVM on the computed features
        self.clf = svm.SVC()
        labels = np.array(train_labels).reshape(len(train_labels), 1)
        hog_features = np.array(hog_features)
        self.clf.fit(hog_features, labels.ravel())


    def predict(self, images):
        """
        Method for image classification of a given set of images
        :param images: Array containing the set of images to be classifier
        :return: Array containing the probabilities for each class, for each input image
        """
        hog_features = []
        # Compute HOG features for all the input images
        for image in images:
            fd, _ = hog(image, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(4, 4),
                        block_norm='L2', visualize=True)
            hog_features.append(fd)
        hog_features = np.array(hog_features)
        # Classification of the feature set
        predicted_classes = self.clf.predict(hog_features)

        return predicted_classes

    def eval(self, test_images, test_labels):
        """
        Performs model evaluation
        :param test_images: Array containig the test images
        :param test_labels: Array containing labels corresponding to the test images
        :return: Classification report
        """
        predicted_classes = self.predict(test_images)
        target_names = ["Class {}".format(i) for i in range(10)]
        class_report = classification_report(test_labels, predicted_classes, target_names=target_names, output_dict=True)
        print(class_report)
        return class_report

    def train_and_eval(self, train_images, train_labels, test_images, test_labels, save_path = ''):
        """
        This method implements a full model training and evaluation routine, and saves all the weights and performance
        metrics, as part of experiment tracking
        :param train_images: Array containing the training images
        :param train_labels: Array containing the training labels
        :param test_images: Array containing the test images
        :param test_labels: Array containing the test labels
        :param save_path: Path to the folder where to store the model weights and metrics
        :return: Dictionary containing performance metrics for the trained model
        """
        self.train(train_images, train_labels)
        report = self.eval(test_images, test_labels)
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

    def save(self, path):
        """
        Saves the model weights
        :param path: Path to the file where to save the model weights
        """
        with open(path, 'wb') as f:
            pickle.dump(self.clf, f)

    def load(self, path):
        """
        Loads a trained model weights
        :param path: Path to the weights file
        """
        with open(path, 'rb') as f:
            self.clf = pickle.load(f)
