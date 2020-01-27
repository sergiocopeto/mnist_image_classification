import unittest
from mnist_image_classification import service

class ServiceTests(unittest.TestCase):

    def test_model_loading_bad_config_file_returns_false(self):
        # Arrange
        config = {'hog_smv_path': 'fake_model_name.h5',
                  'cnn_4layer_path': 'fake_model_name.h5',
                  'vgg_19_path': 'fake_model_name.h5'}
        # Act
        result = service.load_models(config['hog_smv_path'], config['cnn_4layer_path'], config['vgg_19_path'])

        # Assert
        self.assertEqual(result, False)


if __name__ == '__main__':
    unittest.main()