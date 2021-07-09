# Based on David Zimmerer's Work
import fnmatch
import os
import random
import shutil
import string
from time import sleep
import warnings
from abc import abstractmethod


import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


from data_aug.bg_wrapper import get_transforms


def init_arg_dicts(*args):
    args = list(args)
    for i, arg in enumerate(args):
        if arg is None:
            args[i] = dict()

    return args



class AbstractAnomalyDataLoader:
    def __init__(self, base_dir, load_args=None, tmp_dir=None, n_items=None):

        if load_args is None:
            load_args = {}
        self.items = self.load_dataset(base_dir=base_dir, **load_args)

        self.base_dir = base_dir
        self.tmp_dir = tmp_dir

        if self.tmp_dir is not None and self.tmp_dir != "" and self.tmp_dir != "None":
            rnd_str = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(15))
            self.tmp_dir = os.path.join(self.tmp_dir, rnd_str)
            # print(self.tmp_dir)
            if not os.path.exists(self.tmp_dir):
                os.mkdir(self.tmp_dir)

        self.data_len = len(self.items)
        if n_items is None:
            self.n_items = self.data_len
        else:
            self.n_items = int(n_items)

    def reshuffle(self):
        print("Reshuffle...")
        random.shuffle(self.items)

    def __len__(self):
        n_items = self.n_items
        return n_items

    def __getitem__(self, item):

        if item >= self.n_items:
            raise StopIteration()

        idx = item % self.data_len

        return self.get_data_by_idx(idx)

    @abstractmethod
    def get_data_by_idx(self, idx):
        pass

    @abstractmethod
    def copy_to_tmp(self, fn_name):
        pass

    @abstractmethod
    def load_dataset(self, base_dir, **load_args):
        pass

class AnomalyDataSet:
    def __init__(self, data_loader, transforms, batch_size=4, num_processes=4, pin_memory=False, drop_last=False,
                 do_reshuffle=True):
        self.data_loader = data_loader

        self.data_loader = data_loader
        self.batch_size = batch_size
        self.do_reshuffle = do_reshuffle
        self.transforms = transforms

        self.augmenter = MultiThreadedDataLoader(self.data_loader, self.transforms, batch_size=batch_size,
                                                 num_processes=num_processes,
                                                 shuffle=do_reshuffle, pin_memory=pin_memory, drop_last=drop_last)

    def __len__(self):
        return len(self.data_loader) // self.batch_size

    def __iter__(self):
        return iter(self.augmenter)

    def __getitem__(self, index):
        item = self.data_loader[index]
        item = self.transforms(**item)
        return item

class WrappedDataset(Dataset):
    def __init__(self, dataset, transforms, add_dim=True):
        self.transforms = transforms
        self.dataset = dataset
        self.add_dim = add_dim


    def __getitem__(self, index):
        item = self.dataset[index]
        item = self.add_dimension(item)
        item = self.transforms(**item)
        item = self.remove_dimension(item)
        return item

    def add_dimension(self, item):
        if self.add_dim and isinstance(item, dict):
            for k, e in item.items():
                if isinstance(e, np.ndarray) or torch.is_tensor(e):
                    item[k] = e[None]
        return item

    def remove_dimension(self, item):
        if self.add_dim and isinstance(item, dict):
            for k, e in item.items():
                # Added due to the double headed data loader
                if isinstance(e, tuple):
                    out = []
                    for data in e:
                        out.append(data[0])
                    item[k] = tuple(out)
                elif isinstance(e, np.ndarray) or torch.is_tensor(e):
                    item[k] = e[0]
        return item

    def __len__(self):
        return len(self.dataset)


class MultiThreadedDataLoader(object):
    def __init__(self, data_loader, transform, num_processes, batch_size, shuffle=True, timeout=0, pin_memory=False,
                 drop_last=False):
        self.transform = transform
        self.timeout = timeout

        self.cntr = 1
        self.ds_wrapper = WrappedDataset(data_loader, transform)

        self.generator = DataLoader(self.ds_wrapper,
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    num_workers=num_processes,
                                    pin_memory=pin_memory,
                                    drop_last=drop_last,
                                    timeout=timeout)

        self.num_processes = num_processes
        self.iter = None

    def __iter__(self):

        return iter(self.generator)

    def __next__(self):

        try:
            return next(self.generator)

        except RuntimeError:
            print("Queue is empty, None returned")
            warnings.warn("Queue is empty, None returned")

            raise StopIteration




def get_brain_dataset(base_dir, mode="train", batch_size=16, n_items=None, pin_memory=False,
                      num_processes=8, drop_last=False, target_size=128, file_pattern='*.npy',
                      add_noise=False, label_slice=None, input_slice=0, mask_type="",
                      slice_offset=0, do_reshuffle=True, only_labeled_slices=None, labeled_threshold=10,
                      rotate=True, elastic_deform=True, rnd_crop=True, color_augment=True, tmp_dir=None, use_npz=False,
                      add_slices=0, add_transforms=(), double_headed=False, transform_type='single', base_train='default'):
    data_loader = BrainDataLoader(base_dir=base_dir,
                                  n_items=n_items, file_pattern=file_pattern,
                                  input_slice=input_slice, label_slice=label_slice, slice_offset=slice_offset,
                                  only_labeled_slices=only_labeled_slices, labeled_threshold=labeled_threshold,
                                  tmp_dir=tmp_dir, use_npz=use_npz, add_slices=add_slices)

    transforms = get_transforms(mode=mode, target_size=target_size,
                                add_noise=add_noise, mask_type=mask_type,
                                rotate=rotate, elastic_deform=elastic_deform, rnd_crop=rnd_crop,
                                color_augment=color_augment, base_train=base_train,
                                add_transforms=add_transforms, double_headed=double_headed, transform_type=transform_type)

    return AnomalyDataSet(data_loader, transforms, batch_size=batch_size, num_processes=num_processes,
                          pin_memory=pin_memory, drop_last=drop_last, do_reshuffle=do_reshuffle)


class BrainDataLoader(AbstractAnomalyDataLoader):
    def __init__(self, base_dir, n_items=None,
                 file_pattern='*.npy', label_slice=None, input_slice=0, slice_offset=0,
                 only_labeled_slices=None, labeled_threshold=10, tmp_dir=None, use_npz=False, add_slices=0):

        super(BrainDataLoader, self).__init__(base_dir=base_dir, tmp_dir=tmp_dir, n_items=n_items,
                                              load_args=dict(
                                                      pattern=file_pattern, slice_offset=slice_offset + add_slices,
                                                      only_labeled_slices=only_labeled_slices, label_slice=label_slice,
                                                      labeled_threshold=labeled_threshold
                                              ))

        self.use_npz = use_npz
        self.input_slice = input_slice
        self.label_slice = label_slice

        self.add_slices = add_slices

    def get_np_file(self, base_name, source_file, target_name, target_name_tmp, target_name_npz, use_npz_local):

        if os.path.exists(target_name):
            numpy_array = np.load(target_name, mmap_mode="r")
        else:
            if os.path.exists(target_name_tmp):
                while not os.path.exists(target_name):
                    sleep(1)
                numpy_array = np.load(target_name, mmap_mode="r")
            else:
                try:
                    shutil.copy2(source_file, target_name_tmp)
                    try:
                        if use_npz_local:
                            with np.load(target_name_tmp) as a:
                                np.save(target_name_npz, a["data"])
                            os.rename(target_name_npz, target_name)
                        else:
                            os.rename(target_name_tmp, target_name)
                    except:
                        print("Somehow could not rename: ", target_name_tmp)
                except:
                    target_name = source_file

                try:
                    numpy_array = np.load(target_name, mmap_mode="r")
                except:
                    shutil.rmtree(target_name, ignore_errors=True)
                    shutil.rmtree(target_name_tmp, ignore_errors=True)
                    shutil.rmtree(target_name_npz, ignore_errors=True)
                    numpy_array = None

        return numpy_array

    def copy_to_tmp(self, fn_name):
        if self.tmp_dir is not None and self.tmp_dir != "" and self.tmp_dir != "None":
            source_file = fn_name

            base_name = os.path.basename(fn_name)
            base_name_tmp = base_name + "_"

            target_name = os.path.join(self.tmp_dir, base_name)
            target_name_tmp = os.path.join(self.tmp_dir, base_name_tmp)

            use_npz_local = False  
            target_name_npz = base_name
            if self.use_npz and os.path.exists(fn_name[:-3] + "npz"):
                source_file = fn_name[:-3] + "npz"
                base_name_npz = base_name[:-3] + "_npz.npy"
                target_name_npz = os.path.join(self.tmp_dir, base_name_npz)
                use_npz_local = True

            numpy_array = None
            xd = 0
            while numpy_array is None and xd < 20:
                numpy_array = self.get_np_file(base_name, source_file, target_name, target_name_tmp,
                                               target_name_npz, use_npz_local)
                xd += 1
            if numpy_array is None:
                numpy_array = np.zeros((200, 100, 1000, 1000))

        else:
            numpy_array = np.load(fn_name, mmap_mode="r")
        return numpy_array

    def get_data_by_idx(self, idx):

        slice_info = self.items[idx]
        fn_name = slice_info[0]
        slice_idx = slice_info[1]

        numpy_array = self.copy_to_tmp(fn_name)
        numpy_slice = numpy_array[self.input_slice, slice_idx - self.add_slices:slice_idx + self.add_slices + 1, ]

        ret_dict = {'data': numpy_slice, 'fnames': fn_name, 'slice_idxs': slice_idx / 200.}

        if self.label_slice is not None:
            label_slice = numpy_array[self.label_slice, slice_idx - self.add_slices:slice_idx + self.add_slices + 1, ]
            ret_dict['seg'] = label_slice

        del numpy_array

        return ret_dict

    @staticmethod
    def load_dataset(base_dir, pattern='*.npy', slice_offset=0, only_labeled_slices=None, label_slice=None,
                     labeled_threshold=10):
        slices = []

        if isinstance(base_dir, str):
            base_dirs = [base_dir]
        elif isinstance(base_dir, (list, tuple)):
            base_dirs = base_dir
        else:
            raise TypeError("base_dir has to be of type str ot a list/ tuple of strings")

        for base_dir in base_dirs:
            for root, dirs, files in os.walk(base_dir):
                for i, filename in enumerate(sorted(fnmatch.filter(files, pattern))):
                    npy_file = os.path.join(root, filename)
                    numpy_array = np.load(npy_file, mmap_mode="r")

                    file_len = numpy_array.shape[1]

                    if only_labeled_slices is None:
                        slices.extend([(npy_file, j) for j in range(slice_offset, file_len - slice_offset)])
                    else:
                        assert label_slice is not None

                        for s_idx in range(slice_offset, numpy_array.shape[1] - slice_offset):
                            pixel_sum = np.sum(numpy_array[label_slice, s_idx] > 0.1)
                            if pixel_sum > labeled_threshold:
                                if only_labeled_slices is True:
                                    slices.append((npy_file, s_idx))
                            elif pixel_sum == 0:
                                if only_labeled_slices is False:
                                    slices.append((npy_file, s_idx))

        return slices

