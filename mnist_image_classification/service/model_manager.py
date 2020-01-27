import urllib.request

import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.transform import resize
from tensorflow import Graph, Session

from mnist_image_classification.models import HogSVM, Cnn4Layer, VGG_19


class ModelManager():
    """
    This class allows us to manage multiple image classification models, as well as handling different image sources
    """
    def __init__(self):
        """
        Class Initialization.
        Performs initialization of all the models, using different Tensorflow sessions so one can use multiple models at
        the same time
        """
        self.hog_svm = HogSVM()
        self.cnn_graph = Graph()
        with self.cnn_graph.as_default():
            self.cnn_session = Session()
            with self.cnn_session.as_default():
                self.cnn_4layer = Cnn4Layer()
        self.vgg19_graph = Graph()
        with self.vgg19_graph.as_default():
            self.vgg_session = Session()
            with self.vgg_session.as_default():
                self.vgg19 = VGG_19()
        self.load_models('output_models/HogSVM_20200127_105836.h5', 'output_models/Cnn4Layer_20200127_110622.h5', 'output_models/VGG_19_20200127_110721.h5')

    def predict(self, image, model):
        """
        Classifies a given image using the specified model
        :param image: Image to be classified
        :param model: Model to be used for classification
        :return:
        """
        result = None
        if model == 'hog_svm':
            result = self.hog_svm.predict(image)
            result = result[0]
        if model == 'cnn_4layer':
            with self.cnn_graph.as_default():
                with self.cnn_session.as_default():
                    result = self.cnn_4layer.predict(image)
                    result = np.argmax(result[0])
        if model == 'vgg19':
            with self.vgg19_graph.as_default():
                with self.vgg_session.as_default():
                    result = self.vgg19.predict(image)
                    result = np.argmax(result[0])
        return result

    def predict_local(self, _image, model):
        """
        Reads an image from a local source, and calls the classification method
        :param _image: Local path to the image to be classified
        :param model: Model to be used for classification
        :return: Predicted class for the source image
        """
        print(_image, model)
        image = imread(_image)
        image = rgb2gray(image)
        image = resize(image, (28, 28))
        image = [image]
        return self.predict(image, model)

    def predict_url(self, _image, model):
        """
        Reads an image from an url source, and calls the classification method
        :param _image: Local path to the image to be classified
        :param model: Model to be used for classification
        :return: Predicted class for the source image
        """
        image = np.asarray(Image.open(urllib.request.urlopen(_image)))
        image = rgb2gray(image)
        image = resize(image, (28, 28))
        image = [image]
        return self.predict(image, model)

    def load_models(self, hog_model, cnn_model, vgg_model):
        """
        Loads the weights for all the image classification models
        :param hog_model: Path to the HOG_SVM model weights file
        :param cnn_model: Path to the CNN_3Layer model weights file
        :param vgg_model: Path to the VGG19 model weights file
        """
        self.hog_svm.load(hog_model)
        with self.cnn_graph.as_default():
            with self.cnn_session.as_default():
                self.cnn_4layer.load(cnn_model)
        with self.vgg19_graph.as_default():
            with self.vgg_session.as_default():
                self.vgg19.load(vgg_model)