import numpy as np
import os

def is_cluster():
    if 'DATASET_LOCATION' in os.environ.keys() and 'EXPERIMENT_LOCATION' in os.environ.keys():
        return True
    else:
        return False
