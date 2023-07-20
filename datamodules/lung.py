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
        print(base_dir)

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
        #print('aqui1')

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

    @abstractmethod
    def get_np_file(self, base_dir, **load_args):
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
        print(item)
        return item

class WrappedDataset(Dataset):
    def __init__(self, dataset, transforms, add_dim=True):
        self.transforms = transforms
        self.dataset = dataset
        self.add_dim = add_dim

    def __getitem__(self, index):
        #print('index', index)
        item = self.dataset[index]

        if len(item['data'].shape)== 3:
            item = self.add_dimension(item) #works like this because I have 50x50x50 and not 1x50x50x50
            item = self.add_dimension(item) #works like this
        elif len(item['data'].shape)== 4:
            item = self.add_dimension(item)  # works like this because I have 50x50x50 and not 1x50x50x50
        item = self.transforms(**item)
        item = self.remove_dimension(item)
        #print('length_item',len(item))
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
        # print('heree')
        # print(xi['data'].shape)
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
def select_max_patches(data_list, max_patches):
    df_select = pd.DataFrame(data_list, index=['patient', 'full_name']).T
    df1 = df_select.groupby('patient')['full_name'].apply(list).reset_index(name='new')
    data_list = []
    random.seed(1)
    for list_name in df1['new'].tolist():
        if len(list_name) < max_patches:
            data_list.extend(list_name)
        else:
            random_listname = random.sample(list_name, max_patches)
            data_list.extend(random_listname)
    return data_list


def get_lung_dataset(base_dir,  mode="train", batch_size=64, n_items=None, pin_memory=False,
                      num_processes=8, drop_last=False, do_reshuffle=True, step='pretext', realworld_dataset = False,
                      patch_size=(50,50,50), elastic_deform = True, rnd_crop = True, rotate = True,
                      num_threads_in_multithreaded = 1, base_train = 'default',  double_headed=False,
                      target_size = (1,50,50,50), input = 'insp', overlap='20', kfold=1, max_patches=None, split_pts=0
                      ):
    print(overlap)
    print(base_dir)
    if step=='train_cnn_latent':
        patches_train = get_list_of_patients(data_folder=base_dir, step='train_cnn_latent', overlap=overlap, kfold = kfold, max_patches=max_patches, realworld_dataset=realworld_dataset, split_pts=split_pts)
        patches_val = get_list_of_patients(data_folder=base_dir, step='eval', overlap=overlap, kfold = kfold, max_patches=max_patches, realworld_dataset=realworld_dataset, split_pts=split_pts)

        patients_train = np.unique([i.split('_')[0] for i in patches_train])
        patients_val = np.unique([i.split('_')[0] for i in patches_val])

        print('train_patients', len(np.unique([i.split('_')[1] for i in patches_train])))
        print('eval_patients', len(np.unique([i.split('_')[1] for i in patches_val])))

        train = [i for i in patches_train if i.split('_')[0] in patients_train]
        val = [i for i in patches_val if i.split('_')[0] in patients_val]

        print('train_patches', len(train))
        print('eval_patches', len(val))


    else:
        patches = get_list_of_patients(data_folder= base_dir, step=step, overlap=overlap, kfold=kfold, max_patches=max_patches, realworld_dataset=realworld_dataset, split_pts=split_pts)
        patients = np.unique([i.replace('_' + i.split('_')[-1], '') for i in patches]) #np.unique([i.split('_')[0] for i in patches])
        train_pat, val_pat = get_split_deterministic(patients, fold=0, num_splits=5, random_state=12345)
        train = [i for i in patches if i.replace('_' + i.split('_')[-1], '') in train_pat]
        val = [i for i in patches if i.replace('_' + i.split('_')[-1], '') in val_pat]

        if max_patches:
            train = [[x.split('_')[0] + '_' + x.split('_')[1] for x in train], train]
            val = [[x.split('_')[0] + '_' + x.split('_')[1] for x in val], val]
            #for evaluation of different methods purpose
            train = select_max_patches(train, max_patches) #this hyperparameter will define how many patches I really need for the final pretext task
            val = select_max_patches(val, max_patches=300) #this is fixed, I don't need to do the evaluation on all patches


    data_loader_train = LungDataLoader(base_dir=base_dir, list_patients=train, n_items=n_items, input = input, overlap= overlap, kfold=kfold, max_patches=max_patches)


    data_loader_val = LungDataLoader(base_dir=base_dir, list_patients=val, n_items=n_items, input = input, overlap=overlap, kfold=kfold, max_patches=max_patches)

    if step=='pretext':
        double_headed = True
    if step=='fitting_GMM' or step=='train_cnn_latent':
        mode='fit' #val

    transforms = get_simclr_pipeline_transform(mode, patch_size, rnd_crop=rnd_crop,
                                                  elastic_deform=elastic_deform,
                                                  rotate=rotate, base_train=base_train, double_headed= double_headed)


    anomaly_train = AnomalyDataSet(data_loader_train, transforms=transforms, batch_size=batch_size, num_processes=num_processes,
                          pin_memory=pin_memory, drop_last=drop_last, do_reshuffle=do_reshuffle)

    anomaly_val = AnomalyDataSet(data_loader_val, transforms=transforms, batch_size=batch_size, num_processes=num_processes,
                          pin_memory=pin_memory, drop_last=drop_last, do_reshuffle=do_reshuffle)

    return anomaly_train, anomaly_val


def get_lung_dataset_withoutSIMCLR(base_dir,  mode="train", batch_size=6, n_items=None, pin_memory=False,
                      num_processes=8, drop_last=False, do_reshuffle=True,
                      patch_size=(50,50,50), elastic_deform = True, rnd_crop = True, rotate = True,
                      num_threads_in_multithreaded = 1, base_train = 'default',  double_headed=False, target_size = (1,50,50,50)
                      ):

    patients = get_list_of_patients(data_folder= base_dir)
    train, val = get_split_deterministic(patients, fold=0, num_splits=5, random_state=12345)


    data_loader_train = LungDataLoader(base_dir=base_dir, list_patients=train, n_items=n_items)


    data_loader_val = LungDataLoader(base_dir=base_dir, list_patients=val, n_items=n_items)

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

def get_lung_dataset_eval(base_dir,  mode="train", batch_size=64, n_items=None, pin_memory=False,
                      num_processes=8, drop_last=False, do_reshuffle=True, step='pretext', realworld_dataset=False,
                      patch_size=(50,50,50), elastic_deform = True, rnd_crop = True, rotate = True,
                      num_threads_in_multithreaded = 1, base_train = 'default',  double_headed=False, target_size = (1,50,50,50),
                           input = 'insp', overlap='20', kfold=1, max_patches=None, split_pts=0,
                      ):

    patients = get_list_of_patients(data_folder= base_dir, step=step, overlap=overlap, kfold=kfold, max_patches=max_patches,
                                    realworld_dataset=realworld_dataset, split_pts=split_pts)#[0:900000] #delete
    patients_unique = np.unique(
        [i.replace('_' + i.split('_')[-1], '') for i in patients])  # np.unique([i.split('_')[0] for i in patches])
    print('eval patients', len(patients_unique))
    print(patients_unique)
    # print(patients_unique[423])
    # print(patients_unique[480])
    # print(patients_unique[829])
    # print(patients_unique[830])
    # print(patients_unique[831])
    # print(patients_unique[840])
    #data_loader = LungDataLoader_eval(base_dir=base_dir, list_patients=patients, n_items=n_items)
    data_loader = LungDataLoader(base_dir=base_dir, list_patients=patients, n_items=n_items, input = input, overlap=overlap, kfold=kfold, max_patches=max_patches)

    transforms = get_simclr_pipeline_transform(mode, patch_size, rnd_crop=rnd_crop,
                                                  elastic_deform=elastic_deform,
                                                  rotate=rotate, base_train='default', double_headed= double_headed)

    anomaly = AnomalyDataSet(data_loader, transforms=transforms, batch_size=batch_size, num_processes=num_processes,
                          pin_memory=pin_memory, drop_last=drop_last, do_reshuffle=do_reshuffle)

    return anomaly



def get_list_of_patients(data_folder, step, overlap: str, kfold: int, max_patches: int, realworld_dataset: bool, split_pts: int):
    """Reads the txt files from data_folder where the lists of patients to use for each step are.
    Pretext: 50% of all COPD + 50% of all healthy
    Fitting GMM: the same 50% of all healthy, but only patches < 1% emphysema
    Evaluation: 25% of all COPD (unseen) + 25% of all healthy (unseen)
    Test set: 25% of all COPD (unseen) + 25% of all healthy (unseen)


    Args:
        data_folder ([str]): [directory of images]
        step 'string': 'pretext'. Defaults to 'pretext'. Options: 'pretext', 'fitting_GMM', 'eval', 'test'
        overlap 'str'. Overlap between patches. Currently 0 and 20 are available
    """


    print('real_world', realworld_dataset)
    if step == 'pretext':
        with open(os.path.join(data_folder[0], 'overlap' + overlap, 'fold' + str(kfold), 'patches_for_pretext.txt'), "r") as fp:
            list_filenames = json.load(fp)
            print(os.path.join(data_folder[0], 'overlap' + overlap, 'fold' + str(kfold), 'patches_for_pretext.txt'))
    elif step == 'fitting_GMM':
        with open(os.path.join(data_folder[0], 'overlap' + overlap, 'fold' + str(kfold), 'patches_for_GMM.txt'), "r") as fp:
            list_filenames = json.load(fp)
    elif step == 'eval':
        with open(os.path.join(data_folder[0], 'overlap' + overlap, 'fold' + str(kfold), 'patches_for_eval.txt'), "r") as fp: #testset
            list_filenames = json.load(fp)

            if split_pts==1:

                weird = ['COPDGene_A', 'COPDGene_B', 'COPDGene_C', 'COPDGene_D', 'COPDGene_E', 'COPDGene_F', 'COPDGene_G',
                     'COPDGene_H']
                list_filenames = [x for x in list_filenames if x.split('_')[0] + '_' + x.split('_')[1][0] in weird]

            elif split_pts==2:

                weird = ['COPDGene_I', 'COPDGene_J', 'COPDGene_K', 'COPDGene_L', 'COPDGene_M', 'COPDGene_N', 'COPDGene_O',
                     'COPDGene_P']
                list_filenames = [x for x in list_filenames if x.split('_')[0] + '_' + x.split('_')[1][0] in weird]

            elif split_pts==3:

                weird = ['COPDGene_Q', 'COPDGene_R', 'COPDGene_S', 'COPDGene_T', 'COPDGene_U', 'COPDGene_V', 'COPDGene_W',
                     'COPDGene_Y', 'COPDGene_X', 'COPDGene_Z']
                list_filenames = [x for x in list_filenames if x.split('_')[0] + '_' + x.split('_')[1][0] in weird]


            else:
                print('No split of pts list')


    elif step == 'test':
        with open(os.path.join(data_folder[0], 'overlap' + overlap, 'fold' + str(kfold), 'patches_for_testset.txt'), "r") as fp:
            list_filenames = json.load(fp)
    elif step == 'train_cnn_latent':
        realworld_dataset = False

        if realworld_dataset:
            print('real_world', realworld_dataset)
            print('helloooo')
            with open(os.path.join(data_folder[0], 'overlap' + overlap, 'fold' + str(kfold),
                                   'patches_for_realworld_pretext.txt'), "r") as fp:
                list_filenames = json.load(fp)
        else:
            print('here')
            with open(os.path.join(data_folder[0], 'overlap' + overlap, 'fold' + str(kfold), 'patches_for_pretext.txt'), "r") as fp:
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


class LungDataLoader(AbstractAnomalyDataLoader):
    def __init__(self, base_dir, list_patients, n_items=None,
                 file_pattern='*.npz', label_slice=None, input_slice=0, slice_offset=0,
                 only_labeled_slices=None, labeled_threshold=10, tmp_dir=None, use_npz=False, add_slices=0, input = 'insp', overlap='0', kfold=5, max_patches=None):

        super(LungDataLoader, self).__init__(base_dir=base_dir, list_patients= list_patients, tmp_dir=tmp_dir, n_items=n_items,
                                              load_args=dict(
                                                  pattern=file_pattern,
                                                  input = input,
                                                  overlap = overlap,
                                                  kfold = kfold,
                                                  max_patches = max_patches
                                              ))

        self.use_npz = use_npz
        self.input_slice = input_slice
        self.label_slice = label_slice

        self.add_slices = add_slices
        self.input = input


    def get_np_file(self, target_name, input):
        #print(target_name)


        if os.path.exists(target_name):
            try:
                numpy_array = np.load(target_name, mmap_mode="r")
                metadata = load_pickle(target_name.replace('.npz', '.pkl'))
            except:
                print('error')
            #print(target_name)
            try:
                if input == 'insp':
                    patches = numpy_array['insp'].astype(float)
                elif input == 'insp_exp_reg':
                    numpy_array_insp = numpy_array['insp'].astype(float)
                    numpy_array_exp = numpy_array['exp'].astype(float)
                    patches = np.stack([numpy_array_insp, numpy_array_exp])
                elif input== 'insp_jacobian':
                    numpy_array_insp = numpy_array['insp'].astype(float)
                    numpy_array_jac = numpy_array['jac'].astype(float)
                    patches = np.stack([numpy_array_insp, numpy_array_jac])
                else:
                    NotImplementedError


            except:
                print('error')


            #patches = numpy_array_insp # removes the dim I had configured for the batch size
            #print('patch_extract for', target_name)

        return patches, metadata

    def get_data_by_idx(self, idx):
        full_path = self.items[idx]
        full_path = os.path.join(os.path.dirname(full_path), os.path.basename(full_path).lower())
        #print('full_path', full_path)

        #fn_name = full_path.split('/')[-2]

        #img_idx = full_path.split('/')[-1].split('_')[0]
        img_idx = os.path.basename(full_path).replace('_' + os.path.basename(full_path).split('_')[-1], '').replace(' ', '')
        #print('img_idx',img_idx)
        #patch_num = full_path.split('/')[-1].split('.')[0].split('_')[1]

        if 'COPDGene' in self.base_dir[0]:
            patient = 'SUBJECT_ID'
            fev_v = 'FEV1_post'
            fvc_v = 'FVC_post'
            gold_v = 'finalGold'
            fev_fvc_v = 'FEV1_FVC_post'
        elif 'cosyconet' in self.base_dir[0]:
            patient = 'patient'
            fev_v = 'FEV1_GLI'
            fvc_v = 'FVC_GLI'
            gold_v = 'GOLD_gli'
            fev_fvc_v = 'FEV_FVC'


        annotation = pd.read_csv(os.path.join(self.base_dir[0], 'COPD_criteria_complete.csv'),
                                 sep=',', converters={patient: lambda x: str(x)}) #insp_jacobian



        if annotation.loc[annotation[patient].str.lower() == img_idx.lower(), 'condition_COPD_GOLD'].values.size == 0:
            print('empty')
            print(patient, img_idx)
        else:
            label = annotation.loc[annotation[patient].str.lower() == img_idx.lower(), 'condition_COPD_GOLD'].values
            label = label.astype(int)

            gold = annotation.loc[annotation[patient].str.lower() == img_idx.lower(), gold_v].values
            gold[gold <0] = 0
            gold = gold[0]

            fev = annotation.loc[annotation[patient].str.lower() == img_idx.lower(), fev_v].values[0]
            fev = fev.astype(float)
            fev_fvc = annotation.loc[annotation[patient].str.lower() == img_idx.lower(), fev_fvc_v].values[0]
            fev_fvc = float(fev_fvc)
            # print('full_path', full_path)
            patch_patient, metadata = self.get_np_file(full_path, self.input)

            metadata[gold_v] = gold
            metadata[fev_v] = fev
            metadata['fev_fvc'] = fev_fvc

            patch_num = metadata['patch_num']

            # print('ret_dict')
            # print(patch_patient)
            # print(label)
            # print(patch_num)
            # print(metadata)


            ret_dict = {'data': patch_patient, 'label': label, 'patient_name': img_idx, 'patch_num': patch_num,'metadata': metadata}
            #print(ret_dict)
            return ret_dict


    @staticmethod
    def load_dataset(base_dir, list_patients, pattern='*.npz', input = 'insp', overlap='0', kfold=1, max_patches=None):


        directories_read = [os.path.join(base_dir[0], 'patches_new_all_overlap' + overlap, sub) + pattern.replace('*','') for sub in list_patients]
        return directories_read




class LungDataLoader_eval(AbstractAnomalyDataLoader):
    def __init__(self, base_dir, list_patients, n_items=None,
                 file_pattern='*.npz', label_slice=None, input_slice=0, slice_offset=0,
                 only_labeled_slices=None, labeled_threshold=10, tmp_dir=None, use_npz=False, add_slices=0, input = 'insp', overlap='0'):

        super(LungDataLoader_eval, self).__init__(base_dir=base_dir, list_patients= list_patients, tmp_dir=tmp_dir, n_items=n_items,
                                              load_args=dict(
                                                  pattern=file_pattern,
                                                  input=input,
                                                  overlap=overlap
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
            #print(target_name)
            try:
                numpy_array = numpy_array['arr_0'].astype(float)
            except:
                print('error')


            #print('patch_extract for', target_name)



        return numpy_array

    def get_data_by_idx(self, idx):

        full_path = self.items[idx]
        fn_name = full_path.split('/')[-2]
        img_idx = full_path.split('/')[-1].split('_')[0]
        patch_num = full_path.split('/')[-1].split('.')[0].split('_')[1]
        patch_num = np.asarray(patch_num).astype(float)




        annotation = pd.read_csv(os.path.join(self.base_dir[0], 'COPD_criteria_complete.csv'),
                                 sep=',', converters={'patient': lambda x: str(x)}) #insp_jacobian


        annotation = annotation[annotation.notna()]
        annotation = annotation.dropna(subset=["condition_COPD_GOLD"])



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
            #print('problem is here')
            #print(ret_dict['data'].shape)
            #print(ret_dict['label'])
            #print(ret_dict['input_img'])
            #print(ret_dict['patient_name'])
            #print('test',label)
            if label == []:
                print('empty')
            print(img_idx)

            print('patch_successfull_for', img_idx)
            #print(ret_dict['label'], ret_dict['patient_name'], ret_dict['data'].shape)
            return ret_dict


    @staticmethod
    def load_dataset(base_dir, list_patients, pattern='*.npz', input = 'insp', overlap='0'):


        directories_read = [os.path.join(base_dir[0], 'patches_new_all_overlap' + overlap, sub) + pattern.replace('*', '') for sub in list_patients]
        # patches_all_overlap0
        return directories_read













