import keras
import numpy as np

def augment_images(images, labels, augementation_factor=1):
    """
    This function performs augmentation for a given set of images
    :param images: Array containing the images to be processed
    :param labels: Array containing labels corresponding to the input images
    :param augementation_factor: Number of augmentations to perform for each image
    :return: Set of augmented images and their corresponding labels
    """
    augmented_image = []
    augmented_image_labels = []

    # For each input image
    for num, image in enumerate(images):

        image = np.reshape(image, (28, 28, 1))

        # Perform "augementation_factor" augmentations
        for i in range(0, augementation_factor):

            # original image:
            augmented_image.append(image)
            augmented_image_labels.append(labels[num])

            # shear
            augmented_image.append(keras.preprocessing.image.random_shear(image, 0.2, row_axis=0, col_axis=1))
            augmented_image_labels.append(labels[num])
            # shift
            augmented_image.append(keras.preprocessing.image.random_shift(image, 0.2, 0.2, row_axis=0, col_axis=1))
            augmented_image_labels.append(labels[num])
            # zoom
            augmented_image.append(keras.preprocessing.image.random_zoom(image, (0.9, 0.9), row_axis=0, col_axis=1))
            augmented_image_labels.append(labels[num])

            # rotate image
            augmented_image.append(keras.preprocessing.image.random_rotation(image, 20))
            augmented_image_labels.append(labels[num])

    # Convert all the dataset to the original image format
    augmented_image = [np.reshape(image,(28,28)) for image in augmented_image]

    return np.array(augmented_image), np.array(augmented_image_labels)
