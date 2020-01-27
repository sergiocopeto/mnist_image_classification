from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization
from keras.models import Sequential

from mnist_image_classification.models.cnn_base import CnnBase


class Cnn4Layer(CnnBase):
    """
    This class implements an image classification neural network using 4 convolutional layers
    """
    def __init__(self):
        super().__init__()
        self.build_model()

    def build_model(self):
        """
        Builds an image classification neural network with 4 convolutional layers
        """
        # Creating a Sequential Model and adding the layers
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())

        self.model.add(Dense(512, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))

        self.model.add(Dense(128, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))

        self.model.add(Dense(10, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
