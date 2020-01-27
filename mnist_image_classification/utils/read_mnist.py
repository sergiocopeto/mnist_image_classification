import idx2numpy

def read_mnist(images_file, labels_file, num_samples=None):
    """
    This function builds a dataset of "num_samples" images, read from the MNIST dataset
    :param images_file: MNIST images file
    :param labels_file: MNIST labels file
    :param num_samples: Number of samples to collect
    :return: Dataset containing images and their corresponding labels
    """
    images = idx2numpy.convert_from_file(images_file)
    labels = idx2numpy.convert_from_file(labels_file)

    if num_samples is not None:
        images = images[:num_samples]
        labels = labels[:num_samples]

    return images, labels