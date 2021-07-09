import os 
import yaml 

## This is important for the Cluster!

from pyutils.py import is_cluster

glob_conf = dict()
script_dir = os.path.dirname(__file__)
file_name = os.path.join(script_dir, "global_config.yml")
trainer_path = os.path.join(script_dir, "trainer_config.yml")
print("Loading Config from: {}".format(file_name))
with open(file_name, 'r') as file:
    glob_conf = yaml.load(file, Loader=yaml.FullLoader)
if 'DATASET_LOCATION' in os.environ.keys() and 'EXPERIMENT_LOCATION' in os.environ.keys():
    glob_conf['datapath'] = os.environ['DATASET_LOCATION']
    glob_conf['logpath'] = os.environ['EXPERIMENT_LOCATION']
print("datapath: \t{}".format(glob_conf['datapath']))
print("logpath: \t{}".format(glob_conf['logpath']))

print("Loading pl.Trainer defaults from: {}".format(trainer_path))
with open(trainer_path, 'r') as file:
    trainer_defaults = yaml.load(file, Loader=yaml.FullLoader)

if is_cluster():
    trainer_defaults['progress_bar_refresh_rate'] =0
