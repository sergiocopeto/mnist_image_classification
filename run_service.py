import json
from argparse import ArgumentParser
from mnist_image_classification import service

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input', required=True)
    args = parser.parse_args()
    with open(args.input,'r') as f:
        config = json.load(f)
    keys = config.keys()
    if 'hog_smv_path' in keys and 'cnn_4layer_path' in keys and 'vgg_19_path' in keys:
        if service.load_models(config['hog_smv_path'], config['cnn_4layer_path'], config['vgg_19_path']):
            service.app.run(debug=False)
        else:
            print('problem loading models')
            exit(1)
