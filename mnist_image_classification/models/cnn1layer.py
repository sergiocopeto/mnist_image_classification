from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential

from mnist_image_classification.models.cnn_base import CnnBase


class Cnn1Layer(CnnBase):
    """
    This class implements a convolutional neural network with single convolutional layer
    """
    def __init__(self):
        """
        Class initialization
        """
        super().__init__()
        self.build_model()

    def build_model(self):
        """
        Builds a image classification neural network with a single convolutinal layer
        """
        self.model = Sequential()
        self.model.add(Conv2D(28, kernel_size=(3, 3), input_shape=(28, 28, 1)))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # Flattening the 2D arrays for fully connected layers
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(10, activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
