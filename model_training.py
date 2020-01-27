"""
This script performs model training for a given set of models and registers all the experiments,
model weights and performance evaluations.
This script takes a configuration file as an argument, where one can detail all the models to train, the location
of the training samples, and where to store all the gathered information
"""
import json
import os
from argparse import ArgumentParser

from sklearn.model_selection import train_test_split

from mnist_image_classification.models import Cnn1Layer
from mnist_image_classification.models import Cnn3Layer
from mnist_image_classification.models import Cnn4Layer
from mnist_image_classification.models import HogSVM
from mnist_image_classification.models import VGG_19
from mnist_image_classification.utils.image_augmentation import augment_images
from mnist_image_classification.utils.read_mnist import read_mnist


def model_training(config):
    """
    This function performs model training and evaluation based on a configuration file.
    :param config:
    :return:
    """
    results = []

    images, labels = read_mnist(config['train_images_path'], config['train_labels_path'], config['number_of_samples'])

    if config['augment_images']:
        print('PERFORMING IMAGE AUGMENTATION')
        images, labels = augment_images(images, labels, augementation_factor=config['number_of_augmentations'])

    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

    if config['train_hog_svm']:
        model = HogSVM()
        training_result = model.train_and_eval(train_images, train_labels, test_images, test_labels, save_path = config['output_models_folder'])
        results.append(training_result)
    if config['train_cnn_1layer']:
        model = Cnn1Layer()
        training_result = model.train_and_eval(train_images, train_labels, test_images, test_labels, _epochs=10, save_path = config['output_models_folder'])
        results.append(training_result)
    if config['train_cnn_3layer']:
        model = Cnn3Layer()
        training_result = model.train_and_eval(train_images, train_labels, test_images, test_labels, _epochs=10, save_path = config['output_models_folder'])
        results.append(training_result)
    if config['train_cnn_4layer']:
        model = Cnn4Layer()
        training_result = model.train_and_eval(train_images, train_labels, test_images, test_labels, _epochs=10, save_path = config['output_models_folder'])
        results.append(training_result)
    if config['train_vgg19']:
        model = VGG_19()
        training_result = model.train_and_eval(train_images, train_labels, test_images, test_labels, _epochs=10, save_path = config['output_models_folder'])
        results.append(training_result)

    # save results
    results_dict = []
    for result in results:
        results_dict.append({'model_name': result['model_name'],
                            'weights_file': result['weights_file'],
                            'accuracy': result['classification_report']['accuracy'],
                            'precision': result['classification_report']['weighted avg']['precision'],
                            'recall': result['classification_report']['weighted avg']['recall'],
                            'f1-score': result['classification_report']['weighted avg']['f1-score'],})
    output_dict = config
    output_dict['training_results'] = results_dict
    with open(config['output_metrics_report'],'w') as out_file:
        json.dump(output_dict,out_file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input', required=True)
    args = parser.parse_args()
    print(args.input)
    with open(args.input,'r') as f:
        config = json.load(f)
    if not os.path.exists(config['output_models_folder']):
        os.makedirs(config['output_models_folder'])
    model_training(config)
