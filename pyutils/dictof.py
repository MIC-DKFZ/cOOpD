from collections.abc import MutableMapping

import numpy as np
import torch


class DictOf(MutableMapping):
    def __init__(self, save_type=np.ndarray, skip_keys=()):
        self.store = dict()
        self.save_type = save_type
        self.initial = False
        self.skip_keys = skip_keys
        if save_type != np.ndarray:
            raise NotImplementedError

    ## Costum Functions

    def insert(self, input):
        if self.initial is False:
            self._initialize(input)
        else:
            self._insert(input)

    def _insert(self, input:dict):
        self.store = self.update_dict(self.store, self.dict_2_numpy(input, skip_keys=self.skip_keys))
        
    def _initialize(self, input:dict):
        self.store = self.dict_2_numpy(input, skip_keys=self.skip_keys)
        self.initial = True

    def lengths(self):
        out_dict = dict()
        for key, val in self.store.items():
            out_dict[key] = len(val)
        return out_dict

    def shapes(self):
        out_dict = dict()
        for key, val in self.store.items():
            out_dict[key] = val.shape
        return out_dict

    ## Dict Functions
    def __getitem__(self, key):
        return self.store[self._keytransform(key)]

    def __setitem__(self, key, value):
        self.store[self._keytransform(key)] = value

    def __delitem__(self, key):
        del self.store[self._keytransform(key)]

    def __iter__(self):
        return iter(self.store)
    
    def __len__(self):
        return len(self.store)
    
    def _keytransform(self, key):
        return key

    @staticmethod
    def update_dict(base_dict, upt_dict):
        """Appends all values from the upt_dict to base_dict according
        to keys in both of them (iterate over base_dict)

        Args:
            base_dict ([type]): [description]
            upt_dict ([type]): [description]

        Returns:
            [type]: [description]
        """
        for key, val in base_dict.items():
            base_dict[key]=np.concatenate([val, upt_dict[key]])
        return base_dict

    @staticmethod 
    def dict_2_numpy(in_dict, skip_keys=[]):
        """Every value is changed to a numpy array if list, tuple or tensor.
        Except the vale is also in the skip_keys

        Args:
            in_dict ([type]): [description]
            skip_keys (list, optional): [description]. Defaults to [].

        Raises:
            NotImplementedError: [description]

        Returns:
            [type]: [description]
        """
        out_dict = dict()
        for key, val in in_dict.items():
            if key not in skip_keys:
                if torch.is_tensor(val):
                    out_dict[key] = val.detach().cpu().numpy()
                elif isinstance(val, (list, tuple)):
                    out_dict[key] = np.array(val)
                elif isinstance(val, dict): #for the metadata case
                    for k, v in val.items():
                        out_dict[key + '_' + k] = np.array(v)
                elif isinstance(val, np.ndarray):
                    continue
                else:
                    raise NotImplementedError
        return out_dict