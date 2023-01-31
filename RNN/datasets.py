import os, glob, copy
import collections
import random, sys, math
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Sampler
from pathlib import Path

# some code borrowed and modified from MIL-Nature-Medicine-2019 paper:
# credit : https://github.com/MSKCC-Computational-Pathology/MIL-nature-medicine-2019
# and : https://github.com/PhilipChicco/MICCAI2020mil/blob/main/research_mil/loaders/datasets.py


def get_dataset_csv(path_csv):
    scores_df = pd.read_csv(path_csv)

    #ret_dict = scores_df.groupby('patient_name')[['patch_num', 'label', 'nll']].apply(lambda g: g.values.tolist()).to_dict()

    label = np.asarray([int(float(i.replace('[', '').replace(']', ''))) for i in scores_df['label']])
    patient_name = scores_df['patient_name'].to_numpy()
    patch_num = np.asarray(['_'.join([str(i),str(j)]) for i,j in zip(scores_df['patient_name'], (scores_df['patch_num']))])
    score = scores_df['nll'].to_numpy()


    ret_dict = {'patient_name': patient_name, 'patch_num': patch_num, 'label': label, 'score': score}

    return ret_dict


class rnndata(Dataset):
    def __init__(self, dir,
                 s,
                 step = 'train_cnn_latent',
                 shuffle=False):
        self.s = s
        self.dir = dir
        self.step = step
        self.shuffle = shuffle
        self.patientIDX = []
        self.patches = []
        self.patientLBL = []
        if self.dir:
            self.info = get_dataset_csv(self.dir)
        else:
            raise ('Provide a dir')



        self.patientnames = self.info['patient_name']
        self.targets = self.info['label']
        self.score_list = self.info['score']
        self.patchnames = self.info['patch_num']

        print('Number of patches: {}'.format(len(self.score_list)))

    def __getitem__(self, index):
        patientIDX = self.patientnames[index]
        target = self.targets[index]
        patch_name = self.patchnames[index]
        #patch_name = self.patches[index]

        if self.shuffle:
            patch_name = random.sample(patch_name, len(patch_name))

        s = min(self.s, len(patch_name))
        out = []
        names_helper = []
        for i in range(s):
            patches = self.info.keys()[self.info.values().index(patch_name[i])]
            patches = self.info['patch_name']
            patches = self.get_file(self.dir, patch_name[i], self.input, self.overlap)
            #print(patch_name[i])

            if self.transform is not None:
                patches = self.transform(patches)

            out.append(patches)
            names_helper.append(patch_name[i])

        return out, self.targets[index], names_helper

    def get_file(self, patch_name, file):
        target_name = os.path.join(dir, 'patches_new_all_overlap' + overlap, patch_name.lower() + '.npz')
        if os.path.exists(target_name):
            try:
                numpy_array = np.load(target_name, mmap_mode="r")
            except:
                print('error')
            try:
                if input == 'insp':
                    patches = numpy_array['insp'].astype(float)
                    patches = np.expand_dims(patches, axis=0)
                elif input == 'insp_exp_reg':
                    numpy_array_insp = numpy_array['insp'].astype(float)
                    numpy_array_exp = numpy_array['exp'].astype(float)
                    patches = np.stack([numpy_array_insp, numpy_array_exp])
                elif input == 'insp_jacobian':
                    numpy_array_insp = numpy_array['insp'].astype(float)
                    numpy_array_jac = numpy_array['jac'].astype(float)
                    patches = np.stack([numpy_array_insp, numpy_array_jac])
                else:
                    NotImplementedError


            except:
                NotImplementedError
        return patches
    def __len__(self):
        return len(self.targets)


if __name__ == '__main__':
    # SANITY CHECK
    "====================================================="
    path = '/home/silvia/Documents/CRADL/logs_cradl/copdgene/pretext/brain/simclr-resnet18/default/17030544/results_plot_0/GMM 1 Comp/lung_val/all_info.csv'
    split = 'val'

    dset = rnndata(dir=path, s=10)
    loader = DataLoader(dset, batch_size=1, num_workers=0, shuffle=False, pin_memory=False)
    for idx, data in enumerate(loader):
       sys.stdout.write('Dry Run : [{}/{}]\r'.format(idx+1, len(loader.dataset)))
    print(len(loader.dataset))


    for idx, data in enumerate(loader):
        print((data[0]).size())

        if idx == 5:
            break