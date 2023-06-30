import os
import yaml

# get package root directory
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

# get project root directory
PROJECT_ROOT_DIR = os.path.dirname(ROOT_DIR)

# load config file
with open(os.path.join(ROOT_DIR, "config_12.yaml"), 'r') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.Loader)