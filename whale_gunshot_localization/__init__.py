import os
import yaml

# get package root directory
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

# load config file
with open(os.path.join(ROOT_DIR, "config.yaml"), 'r') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.Loader)