# Based on David Zimmerer's Work
import fnmatch
import json
import os
import random
import shutil
import string
from time import sleep
import warnings
from abc import abstractmethod


import numpy as np
import torch
from batchgenerators.dataloading.data_loader import DataLoader


from data_aug.bg_wrapper import get_transforms, get_simclr_pipeline_transform

from batchgenerators.utilities.file_and_folder_operations import *
from batchgenerators.utilities.data_splitting import get_split_deterministic
from batchgenerators.dataloading import MultiThreadedAugmenter, SingleThreadedAugmenter

import pandas as pd

from datamodules.extract_patches import extract_patches_3d_fromMask, extract_allpatches_3d_fromMask, getN_allpatches_3d_fromMask

from torch.utils.data import DataLoader, Dataset

import os


class AbstractAnomalyDataLoader:
    def __init__(self, base_dir, list_patients, load_args=None, tmp_dir=None, n_items=None):

        if load_args is None:
            load_args = {}
        self.items = self.load_dataset(base_dir=base_dir, list_patients=list_patients, **load_args)
        print('load is ok')

        self.base_dir = base_dir
        self.list_patients = list_patients
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
    def load_dataset(self, base_dir, list_patients, **load_args):
        pass


class AnomalyDataSet:
    def __init__(self, data_loader, transforms, batch_size=64, num_processes=4, pin_memory=False, drop_last=False,
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
        print(len(self.data_loader))
        print(self.batch_size)
        return len(self.data_loader) // self.batch_size

    def __iter__(self):
        return iter(self.augmenter)

    def __getitem__(self, index):
        item = self.data_loader[index]
        item = self.transforms(**item)
        return item

# class AnomalyDataSet:
#     def __init__(self, data_loader, transforms, batch_size=4, num_processes=4, pin_memory=False, drop_last=False,
#                  do_reshuffle=True):
#         self.data_loader = data_loader
#
#         self.data_loader = data_loader
#         self.batch_size = batch_size
#         self.do_reshuffle = do_reshuffle
#         self.transforms = transforms
#
#         self.augmenter = MultiThreadedAugmenter(self.data_loader, self.transforms, num_processes=num_processes,
#                                         num_cached_per_queue=1,
#                                         seeds=None, pin_memory=False)
#
#
#     def __len__(self):
#         #CHECK
#         print(len(self.data_loader.indices))
#         print(self.batch_size)
#         print(len(self.data_loader.indices) // self.batch_size)
#         #return len(self.data_loader.indices) // self.batch_size
#
#         return len(self.data_loader.indices)
#
#     def __iter__(self):
#         return iter(self.augmenter)
#
#     def __getitem__(self, index):
#         item = self.data_loader[index]
#         item = self.transforms(**item)
#         return item
class WrappedDataset(Dataset):
    def __init__(self, dataset, transforms, add_dim=True):
        self.transforms = transforms
        self.dataset = dataset
        self.add_dim = add_dim

    def __getitem__(self, index):
        item = self.dataset[index]
        print('index_name', item['patient_name'], item['data'].shape)
        item = self.add_dimension(item) #works like this because I have 50x50x50 and not 1x50x50x50
        item = self.add_dimension(item) #works like this
        item = self.transforms(**item)
        item = self.remove_dimension(item)
        print('length_item',len(item))
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


def init_arg_dicts(*args):
    args = list(args)
    for i, arg in enumerate(args):
        if arg is None:
            args[i] = dict()

    return args


class SimCLRDataTransform(object):
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, **data_dict):
        xi = self.transform(**data_dict)
        xj = self.transform(**data_dict)
        xi['data'] = (xi['data'], xj['data'])
        return xi


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



def get_brain_dataset(base_dir,  mode="train", batch_size=64, n_items=None, pin_memory=False,
                      num_processes=8, drop_last=False, do_reshuffle=True, step='pretext',
                      patch_size=(50,50,50), elastic_deform = True, rnd_crop = True, rotate = True,
                      num_threads_in_multithreaded = 1, base_train = 'default',  double_headed=False, target_size = (1,50,50,50)
                      ):

    patients = get_list_of_patients(data_folder= base_dir, step=step)
    train, val = get_split_deterministic(patients, fold=0, num_splits=5, random_state=12345)
    # dataloader_train = DataLoader3D(train, batch_size, patch_size, num_threads_in_multithreaded)
    # dataloader_validation = DataLoader3D(val, batch_size, patch_size, num_threads_in_multithreaded)
    # tr_transforms = get_simclr_pipeline_transform(mode, patch_size, rnd_crop = rnd_crop, elastic_deform = elastic_deform,
    #                                               rotate = rotate, base_train = 'default')
    #
    # tr_gen = AnomalyDataSet(data_loader=dataloader_train, transforms= SimCLRDataTransform(tr_transforms), batch_size=batch_size)
    # val_gen = AnomalyDataSet(data_loader=dataloader_validation, transforms= SimCLRDataTransform(tr_transforms), batch_size=batch_size)


    data_loader_train = BrainDataLoader(base_dir=base_dir, list_patients=train, n_items=n_items)


    data_loader_val = BrainDataLoader(base_dir=base_dir, list_patients=val, n_items=n_items)

    if step=='pretext':
        double_headed = True
    if step=='fitting_GMM':
        mode='fit' #val

    transforms = get_simclr_pipeline_transform(mode, patch_size, rnd_crop=rnd_crop,
                                                  elastic_deform=elastic_deform,
                                                  rotate=rotate, base_train='default', double_headed= double_headed)

    # transforms_val = get_simclr_pipeline_transform(mode, patch_size, rnd_crop=rnd_crop,
    #                                               elastic_deform=elastic_deform,
    #                                               rotate=rotate, base_train='default', double_headed= double_headed)

    # transforms = get_transforms(mode=mode, target_size=target_size, rotate=rotate, elastic_deform=elastic_deform, rnd_crop=rnd_crop,
    #                             base_train=base_train, double_headed=double_headed)

    #transforms=SimCLRDataTransform(transforms_train)
    #transforms=SimCLRDataTransform(transforms_val)

    anomaly_train = AnomalyDataSet(data_loader_train, transforms=transforms, batch_size=batch_size, num_processes=num_processes,
                          pin_memory=pin_memory, drop_last=drop_last, do_reshuffle=do_reshuffle)

    anomaly_val = AnomalyDataSet(data_loader_val, transforms=transforms, batch_size=batch_size, num_processes=num_processes,
                          pin_memory=pin_memory, drop_last=drop_last, do_reshuffle=do_reshuffle)

    return anomaly_train, anomaly_val


def get_brain_dataset_withoutSIMCLR(base_dir,  mode="train", batch_size=6, n_items=None, pin_memory=False,
                      num_processes=8, drop_last=False, do_reshuffle=True,
                      patch_size=(50,50,50), elastic_deform = True, rnd_crop = True, rotate = True,
                      num_threads_in_multithreaded = 1, base_train = 'default',  double_headed=False, target_size = (1,50,50,50)
                      ):

    patients = get_list_of_patients(data_folder= base_dir)
    train, val = get_split_deterministic(patients, fold=0, num_splits=5, random_state=12345)


    data_loader_train = BrainDataLoader(base_dir=base_dir, list_patients=train, n_items=n_items)


    data_loader_val = BrainDataLoader(base_dir=base_dir, list_patients=val, n_items=n_items)

    transforms_train = get_simclr_pipeline_transform(mode, patch_size, rnd_crop=rnd_crop,
                                                  elastic_deform=elastic_deform,
                                                  rotate=rotate, base_train='default')

    transforms_val = get_simclr_pipeline_transform('val', patch_size, rnd_crop=rnd_crop,
                                                  elastic_deform=elastic_deform,
                                                  rotate=rotate, base_train='default')

    anomaly_train = AnomalyDataSet(data_loader_train, transforms=transforms_train, batch_size=batch_size, num_processes=num_processes,
                          pin_memory=pin_memory, drop_last=drop_last, do_reshuffle=do_reshuffle)

    anomaly_val = AnomalyDataSet(data_loader_val, transforms=transforms_val, batch_size=batch_size, num_processes=num_processes,
                          pin_memory=pin_memory, drop_last=drop_last, do_reshuffle=do_reshuffle)

    #return data_loader_train, data_loader_val
    return anomaly_train, anomaly_val

def get_brain_dataset_eval(base_dir,  mode="train", batch_size=64, n_items=None, pin_memory=False,
                      num_processes=8, drop_last=False, do_reshuffle=True, step='pretext',
                      patch_size=(50,50,50), elastic_deform = True, rnd_crop = True, rotate = True,
                      num_threads_in_multithreaded = 1, base_train = 'default',  double_headed=False, target_size = (1,50,50,50)
                      ):

    patients = get_list_of_patients(data_folder= base_dir, step=step)
    #data_loader = BrainDataLoader_eval(base_dir=base_dir, list_patients=patients, n_items=n_items)
    data_loader = BrainDataLoader(base_dir=base_dir, list_patients=patients, n_items=n_items)

    transforms = get_simclr_pipeline_transform(mode, patch_size, rnd_crop=rnd_crop,
                                                  elastic_deform=elastic_deform,
                                                  rotate=rotate, base_train='default', double_headed= double_headed)

    anomaly = AnomalyDataSet(data_loader, transforms=transforms, batch_size=batch_size, num_processes=num_processes,
                          pin_memory=pin_memory, drop_last=drop_last, do_reshuffle=do_reshuffle)

    return anomaly


def get_list_of_patients(data_folder, step):
    """Reads the txt files from data_folder where the lists of patients to use for each step are.
    Pretext: 50% of all COPD + 50% of all healthy
    Fitting GMM: the same 50% of all healthy
    Evaluation: 25% of all COPD (unseen) + 5% of all healthy (unseen)
    Test set: 25% of all COPD (unseen) + 5% of all healthy (unseen)


    Args:
        data_folder ([str]): [directory of images]
        step 'string': 'pretext'. Defaults to 'pretext'. Options: 'pretext', 'fitting_GMM', 'eval', 'test'.
    """
    # npy_files = subfiles(data_folder[0], suffix=".npz", join=True)
    # # remove npy file extension
    # patients = [str(os.path.basename(i)).split('.')[0] for i in npy_files]
    #
    # annotation = pd.read_csv(
    #     os.path.join(data_folder[0].replace('/pre-processed/no_resample', ''), 'COPD_criteria_complete.csv'),
    #     sep=',', converters={'patient': lambda x: str(x)})
    #
    # annotation = annotation[annotation.notna()]
    # annotation = annotation.dropna(subset=["condition_COPD_GOLD"])
    # list_patients_low = [x.lower() for x in patients]
    # dir_csv = annotation['patient'].to_list()
    # dir_csv = [x.lower() for x in dir_csv]
    # print('attention these npz files dont have labels:', list(set(list_patients_low).difference(dir_csv)))
    # print('move npz file')
    #
    # healthy = annotation.loc[annotation['condition_COPD_GOLD'] == 0]
    # COPD = annotation.loc[annotation['condition_COPD_GOLD'] == 1]
    #
    # dir_csv_healthy = healthy['patient'].to_list()
    # dir_csv_healthy = [x.lower() for x in dir_csv_healthy]
    #
    # print(list(set(list_patients_low).intersection(dir_csv_healthy)))
    # print(len(list(set(list_patients_low).intersection(dir_csv_healthy))))
    #
    # healthy_list = list(set(list_patients_low).intersection(dir_csv_healthy))
    #
    # dir_csv_copd = COPD['patient'].to_list()
    # dir_csv_copd = [x.lower() for x in dir_csv_copd]
    #
    # print(list(set(list_patients_low).intersection(dir_csv_copd)))
    # print(len(list(set(list_patients_low).intersection(dir_csv_copd))))
    #
    # copd_list = list(set(list_patients_low).intersection(dir_csv_copd))
    #
    # print('final for training')
    # print('num_copd_random', len(random.choices(copd_list, k=200)))
    # print('num_healthy', len(healthy_list))
    #
    # filenames_for_pretext = random.choices(copd_list, k=200) + healthy_list
    #
    # return filenames_for_pretext


    # if step == 'pretext':
    #     with open(os.path.join(data_folder[0], 'filenames_for_pretext.txt'), "r") as fp, \
    #             open(os.path.join(data_folder[0], 'filenames_copd.txt'), "r") as copd, \
    #             open(os.path.join(data_folder[0], 'filenames_healthy.txt'), "r") as healthy:
    #         filenames = json.load(fp)
    #         copd_filenames = json.load(copd)
    #         healthy_filenames = json.load(healthy)
    #         list_filenames = extract_fixedN_patches(data_folder, filenames, copd_filenames, healthy_filenames, num_patches=50)
    # elif step == 'fitting_GMM':
    #     with open(os.path.join(data_folder[0], 'filenames_for_GMM.txt'), "r") as fp:
    #         filenames = json.load(fp)
    #         list_filenames = extract_allavailable_patches(data_folder, filenames)
    # elif step == 'eval':
    #     with open(os.path.join(data_folder[0], 'filenames_for_eval.txt'), "r") as fp:
    #         filenames = json.load(fp)
    #     #filenames = os.listdir(os.path.join(data_folder[0], 'patches_insp_overlap0'))
    #     list_filenames = extract_allavailable_patches(data_folder, filenames)
    # elif step == 'test':
    #     with open(os.path.join(data_folder[0], 'filenames_for_testset.txt'), "r") as fp:
    #         filenames = json.load(fp)
    #         list_filenames = extract_allavailable_patches(data_folder, filenames)
    # else:
    #     raise NotImplementedError
    #
    # return list_filenames


    if step == 'pretext':
        with open(os.path.join(data_folder[0], 'patches_for_pretext.txt'), "r") as fp:
            list_filenames = json.load(fp)
    elif step == 'fitting_GMM':
        with open(os.path.join(data_folder[0], 'patches_for_GMM.txt'), "r") as fp:
            list_filenames = json.load(fp)
    elif step == 'eval':
        with open(os.path.join(data_folder[0], 'patches_for_eval.txt'), "r") as fp:
            list_filenames = json.load(fp)
    elif step == 'test':
        with open(os.path.join(data_folder[0], 'patches_for_testset.txt'), "r") as fp:
            list_filenames = json.load(fp)
    else:
        raise NotImplementedError

    return list_filenames



def extract_allavailable_patches(data_folder, filenames):
    #If I want to extract all patches
    all_patches = [_ for _ in os.listdir(os.path.join(data_folder[0], 'patches_all_overlap0')) if _.endswith('.npz')]
    list_filenames = []
    for sub in all_patches:
        for s in filenames:
            if sub.split('_')[0] == s:
                print(s)
                list_filenames.append(sub.split('.')[0])
    return list_filenames

def extract_fixedN_patches(data_folder, filenames, copd_filenames, healthy_filenames, num_patches):
    #If I want a fixed number of patches to be extracted:
    partial = []
    #all_patches_organized = sorted([_.split('.')[0] for _ in os.listdir(os.path.join(data_folder[0], 'patches_all_overlap0')) if _.endswith('.npz')])

    #choose the ones that have > 20% emphysema
    all_meta_organized = sorted([_ for _ in os.listdir(os.path.join(data_folder[0], 'patches_all_overlap0')) if _.endswith('.pkl')])
    all_patches_copd= sorted([file.split('.')[0] for file in all_meta_organized
                                    if (load_pickle(os.path.join(data_folder[0], 'patches_all_overlap0', file))['emphysema_perc'] >0.2 and file.split('_')[0] in copd_filenames)])
    all_patches_healthy= sorted([file.split('.')[0] for file in all_meta_organized if file.split('_')[0] in healthy_filenames])
    all_patches_organized = random.sample(all_patches_copd + all_patches_healthy, len(all_patches_copd + all_patches_healthy))


    #num_patches = 50 #num of patches to extract per image
    for key in filenames:
        filenames_final = []
        for element in all_patches_organized:
            if element.startswith(key):
                filenames_final.append(element)
        if not filenames_final:
            pass
        elif len(filenames_final) >= num_patches:
            partial.append(random.sample(filenames_final, num_patches))
        elif len(filenames_final) < num_patches:
            partial.append(random.sample(filenames_final, len(filenames_final)))
    print('num_patients',len(partial))
    list_filenames = [item for sublist in partial for item in sublist]
    print('num_patches',len(list_filenames))
    return random.sample(list_filenames, len(list_filenames))

#I'm not using this class
# class DataLoader3D(DataLoader):
#     def __init__(self, data, batch_size, patch_size, num_threads_in_multithreaded, seed_for_shuffle=1234,
#                  return_incomplete=False, shuffle=True, infinite=True):
#         """
#         data must be a list of patients as returned by get_list_of_patients (and split by get_split_deterministic)
#         patch_size is the spatial size the retured batch will have
#         """
#         super().__init__(data, batch_size, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete, shuffle,
#                          infinite)
#         self.patch_size = patch_size
#         self.num_modalities = 1 #change
#         self.indices = list(range(len(data)))
#         self.base_dir = '/home/silvia/Documents/CRADL/pre-processed/insp/'
#
#     def load_patient(self, path_imgs, path_labels, patch_dim):
#         data_npz = np.load(path_imgs + ".npz", mmap_mode="r")
#         data_img = data_npz[data_npz.files[0]]
#         data_label_npz = np.load(path_labels + ".npz", mmap_mode="r")
#         data_label = data_label_npz[data_label_npz.files[0]]
#
#         patches = extract_patches_3d_fromMask(data_img, data_label, patch_dim, max_patches=1, random_state=12345)
#
#         return patches
#
#     def generate_train_batch(self):
#         idx = self.get_indices()
#         patients_for_batch = [self._data[i] for i in idx]
#
#         patches = np.zeros((self.batch_size, self.num_modalities, *self.patch_size), dtype=np.float32)
#
#         metadata = []
#         patient_names = []
#
#         print('here')
#
#         # iterate over patients_for_batch and include them in the batch
#         for i, j in enumerate(patients_for_batch):
#
#
#             patient_data = self.load_patient(os.path.join(self.base_dir, j),
#                                              os.path.join(self.base_dir.replace('insp', 'labels'), j),
#                                              patch_dim= (self.num_modalities,) + self.patch_size)
#
#
#             patches[i] = patient_data[0]
#
#             patient_names.append(j)
#
#         ret_dict = {'data': patches, 'names': patient_names}
#
#         return ret_dict


class BrainDataLoader(AbstractAnomalyDataLoader):
    def __init__(self, base_dir, list_patients, n_items=None,
                 file_pattern='*.npz', label_slice=None, input_slice=0, slice_offset=0,
                 only_labeled_slices=None, labeled_threshold=10, tmp_dir=None, use_npz=False, add_slices=0):

        super(BrainDataLoader, self).__init__(base_dir=base_dir, list_patients= list_patients, tmp_dir=tmp_dir, n_items=n_items,
                                              load_args=dict(
                                                  pattern=file_pattern
                                              ))

        self.use_npz = use_npz
        self.input_slice = input_slice
        self.label_slice = label_slice

        self.add_slices = add_slices

    def get_np_file(self, target_name, folder_name):

        if os.path.exists(target_name):
            try:
                numpy_array = np.load(target_name, mmap_mode="r")
                metadata = load_pickle(target_name.replace('.npz', '.pkl'))
            except:
                print('error')
            print(target_name)
            try:
                numpy_array_insp = numpy_array['insp'].astype(float)
            except:
                print('error')
            try:
                numpy_array_label = numpy_array['label'].astype(float)
            except:
                print('error')

            #numpy_array_insp = numpy_array[numpy_array.files['insp']].astype(float)
            #numpy_array_label = numpy_array[numpy_array.files['label']].astype(float)

            # # import SimpleITK as sitk
            # # sitk.WriteImage(sitk.GetImageFromArray(numpy_array), '/home/silvia/Downloads/' + os.path.basename(target_name)[:-3] + 'nii.gz')
            # if os.path.exists(target_name.replace(folder_name, 'labels')):
            #     numpy_label = np.load(target_name.replace(folder_name, 'labels'), mmap_mode="r")
            #     numpy_label = numpy_label[numpy_label.files[0]].astype(float)
            # else:
            #     print('label_not')


            #if reading normal image

            # patch_dim = (1,50,50,50) #change
            # print('trying for', target_name)
            # patches = extract_patches_3d_fromMask(numpy_array_insp, numpy_array_label, patch_dim, max_patches=1, random_state=12345) #max_patches=100000
            # patches = patches[0,:,:,:,:] # removes the dim I had configured for the batch size
            # print('patch_extract for', target_name)

            #if reading patch
            patches = numpy_array_insp # removes the dim I had configured for the batch size
            print('patch_extract for', target_name)

        return patches, metadata

    def get_data_by_idx(self, idx):
        list_strange = ['052503978', '045682744', '003258511', '10925607x']

        full_path = self.items[idx]
        fn_name = full_path.split('/')[-2]

        img_idx = full_path.split('/')[-1].split('_')[0]
        patch_num = full_path.split('/')[-1].split('.')[0].split('_')[1]

        if img_idx in list_strange:
            print(img_idx, 'strange might fail')



        #I have to put this somewhere else. where??
        annotation = pd.read_csv(os.path.join(self.base_dir[0].replace('/pre-processed/all_no_resample',''), 'COPD_criteria_complete.csv'),
                                 sep=',', converters={'patient': lambda x: str(x)}) #insp_jacobian

        #drop missing values
        #annotation = annotation.dropna(subset=['condition_COPD_GOLD'])
        #print(annotation['condition_COPD_GOLD'])
        annotation = annotation[annotation.notna()]
        annotation = annotation.dropna(subset=["condition_COPD_GOLD"])

        # pd.set_option("display.max_rows", None, "display.max_columns", None)
        # print(annotation['patient'], annotation['condition_COPD_GOLD'])

        print(annotation.loc[annotation['patient'].str.lower() == img_idx.lower(), 'condition_COPD_GOLD'].values)
        print(img_idx+'.nii.gz')

        if annotation.loc[annotation['patient'].str.lower() == img_idx.lower(), 'condition_COPD_GOLD'].values == []:
            print('empty')
        else:
            label = annotation.loc[annotation['patient'].str.lower() == img_idx.lower(), 'condition_COPD_GOLD'].values
            label = label.astype(int)


            patch_patient, metadata = self.get_np_file(full_path, fn_name)

            ret_dict = {'data': patch_patient, 'label': label, 'patient_name': img_idx,
                        'patch_num': patch_num, 'metadata': metadata} #'input_img': fn_name,
            print('problem is here')
            print(ret_dict['data'].shape)
            print(ret_dict['label'])
            #print(ret_dict['input_img'])
            print(ret_dict['patient_name'])
            #print('test',label)
            if label == []:
                print('empty')
            print(img_idx)

            # try:
            #     from batchviewer import view_batch
            #     # same patient, two augm
            #     # first pair
            #     view_batch(ret_dict['data'])
            #
            #
            # except ImportError:
            #     view_batch = None
            print('patch_successfull_for', img_idx)
            print(ret_dict['label'], ret_dict['patient_name'], ret_dict['data'].shape)
            return ret_dict


    @staticmethod
    def load_dataset(base_dir, list_patients, pattern='*.npz'):
        # print(base_dir)
        # print(list_patients, len(list_patients))
        # directories = []
        # for patient in list_patients:
        #     directories.append(os.path.join(base_dir[0], patient)+pattern.replace('*', ''))
        #
        # annotation = pd.read_csv(os.path.join(base_dir[0].replace('/pre-processed/no_resample',''), 'COPD_criteria_complete.csv'),
        #                          sep=',', converters={'patient': lambda x: str(x)})
        #
        # annotation = annotation[annotation.notna()]
        # annotation = annotation.dropna(subset=["condition_COPD_GOLD"])
        # dir_folder = [os.path.basename(i).split('.')[0] for i in directories]
        # #print(dir_folder)
        # dir_csv = annotation['patient'].to_list()
        # dir_csv = [x.lower() for x in dir_csv]
        # dir_folder = [x.lower() for x in dir_folder]
        # # print(dir_csv)
        # # print(set(dir_folder).difference(dir_csv))
        # print('attention these npz files dont have labels:', list(set(dir_folder).difference(dir_csv)))
        # print('move npz file')
        # # for name_to_remove in list(set(dir_folder).difference(dir_csv)):
        # #     #print(name_to_remove)
        # #     dir_folder.remove(name_to_remove)
        # #
        # # print(set(dir_folder).difference(dir_csv))
        # directories = [base_dir[0] + '/' + sub + '.npz' for sub in dir_folder]
        # print('true training/test', len(directories))
        #
        # healthy = annotation.loc[annotation['condition_COPD_GOLD'] == 0]
        # COPD = annotation.loc[annotation['condition_COPD_GOLD'] == 1]
        #
        # # final_patient = []
        # # for patient in directories:
        # #     print(os.path.basename(patient).split('.')[0])
        # #     final_patient.append(os.path.basename(patient).split('.')[0])
        # final_patient = [os.path.basename(patient).split('.')[0] for patient in directories]
        # print(final_patient)
        # healthy[healthy['patient'].isin(final_patient)]
        #
        #
        #
        # print(healthy)
        # print(COPD)
        #
        #
        # #directories_for_pretext = random.choices(directories, k=)
        #
        #
        # return directories



        # remove npy file extension
        # patients = [str(os.path.basename(i)).split('.')[0] for i in npy_files]
        # directory_patient = [i for i in npy_files]

        #return directory_patient

        ##old

        # annotation = pd.read_csv(os.path.join(base_dir[0].replace('/pre-processed/no_resample', ''), 'COPD_criteria_complete.csv'),
        #                          sep=',', converters={'patient': lambda x: str(x)})
        #
        # annotation = annotation[annotation.notna()]
        # annotation = annotation.dropna(subset=["condition_COPD_GOLD"])
        # list_patients_low = [x.lower() for x in list_patients]
        # dir_csv = annotation['patient'].to_list()
        # dir_csv = [x.lower() for x in dir_csv]
        # print('attention these npz files dont have labels:', list(set(list_patients_low).difference(dir_csv)))
        # print('move npz file')
        #
        # healthy = annotation.loc[annotation['condition_COPD_GOLD'] == 0]
        # COPD = annotation.loc[annotation['condition_COPD_GOLD'] == 1]
        #
        # dir_csv_healthy = healthy['patient'].to_list()
        # dir_csv_healthy = [x.lower() for x in dir_csv_healthy]
        #
        # print(list(set(list_patients_low).intersection(dir_csv_healthy)))
        # print(len(list(set(list_patients_low).intersection(dir_csv_healthy))))
        #
        # healthy_list = list(set(list_patients_low).intersection(dir_csv_healthy))
        #
        # dir_csv_copd = COPD['patient'].to_list()
        # dir_csv_copd = [x.lower() for x in dir_csv_copd]
        #
        # print(list(set(list_patients_low).intersection(dir_csv_copd)))
        # print(len(list(set(list_patients_low).intersection(dir_csv_copd))))
        #
        # copd_list = list(set(list_patients_low).intersection(dir_csv_copd))
        #
        # directories_healthy = [base_dir[0] + '/' + sub + '.npz' for sub in healthy_list]
        # directories_copd = [base_dir[0] + '/' + sub + '.npz' for sub in copd_list]
        # print('final for training')
        # print('num_copd_random',len(random.choices(directories_copd, k=200)))
        # print('num_healthy',len(directories_healthy))
        #
        #
        # directories_for_pretext = random.choices(directories_copd, k=200) + directories_healthy
        #
        # return directories_for_pretext

        ##new
        #directories_read = [base_dir[0] + '/' + sub + pattern.replace('*','') for sub in list_patients]


        directories_read = [os.path.join(base_dir[0], 'patches_new_all_overlap0', sub) + pattern.replace('*','') for sub in list_patients]
        #patches_all_overlap0
        return directories_read




class BrainDataLoader_eval(AbstractAnomalyDataLoader):
    def __init__(self, base_dir, list_patients, n_items=None,
                 file_pattern='*.npz', label_slice=None, input_slice=0, slice_offset=0,
                 only_labeled_slices=None, labeled_threshold=10, tmp_dir=None, use_npz=False, add_slices=0):

        super(BrainDataLoader_eval, self).__init__(base_dir=base_dir, list_patients= list_patients, tmp_dir=tmp_dir, n_items=n_items,
                                              load_args=dict(
                                                  pattern=file_pattern
                                              ))

        self.use_npz = use_npz
        self.input_slice = input_slice
        self.label_slice = label_slice

        self.add_slices = add_slices

    def get_np_file(self, target_name, folder_name):

        if os.path.exists(target_name):
            try:
                numpy_array = np.load(target_name, mmap_mode="r")
            except:
                print('error')
            print(target_name)
            try:
                numpy_array = numpy_array['arr_0'].astype(float)
            except:
                print('error')


            print('patch_extract for', target_name)



        return numpy_array

    def get_data_by_idx(self, idx):

        full_path = self.items[idx]
        fn_name = full_path.split('/')[-2]
        img_idx = full_path.split('/')[-1].split('_')[0]
        patch_num = full_path.split('/')[-1].split('.')[0].split('_')[1]
        patch_num = np.asarray(patch_num).astype(float)




        #I have to put this somewhere else. where??
        annotation = pd.read_csv(os.path.join(self.base_dir[0].replace('/pre-processed/no_resample',''), 'COPD_criteria_complete.csv'),
                                 sep=',', converters={'patient': lambda x: str(x)}) #insp_jacobian

        #drop missing values
        #annotation = annotation.dropna(subset=['condition_COPD_GOLD'])
        #print(annotation['condition_COPD_GOLD'])
        annotation = annotation[annotation.notna()]
        annotation = annotation.dropna(subset=["condition_COPD_GOLD"])

        # pd.set_option("display.max_rows", None, "display.max_columns", None)
        # print(annotation['patient'], annotation['condition_COPD_GOLD'])

        print(annotation.loc[annotation['patient'].str.lower() == img_idx.lower(), 'condition_COPD_GOLD'].values)
        print(img_idx+'.nii.gz')

        if annotation.loc[annotation['patient'].str.lower() == img_idx.lower(), 'condition_COPD_GOLD'].values == []:
            print('empty')
        else:
            label = annotation.loc[annotation['patient'].str.lower() == img_idx.lower(), 'condition_COPD_GOLD'].values
            label = label.astype(int)


            list_patches_patient = self.get_np_file(full_path, fn_name)
            print(len(list_patches_patient))

            ret_dict = {'data': list_patches_patient, 'label': label, 'patient_name': img_idx, 'patch_num': patch_num} #'input_img': fn_name,
            print('problem is here')
            print(ret_dict['data'].shape)
            print(ret_dict['label'])
            #print(ret_dict['input_img'])
            print(ret_dict['patient_name'])
            #print('test',label)
            if label == []:
                print('empty')
            print(img_idx)

            print('patch_successfull_for', img_idx)
            print(ret_dict['label'], ret_dict['patient_name'], ret_dict['data'].shape)
            return ret_dict


    @staticmethod
    def load_dataset(base_dir, list_patients, pattern='*.npz'):
        directories_read = [os.path.join(base_dir[0], 'patches_insp_overlap0') + '/' + sub for sub in list_patients]

        #just for experiment
        directories_read = directories_read[:1000]

        return directories_read













