#!/usr/bin/env python

import os
import pandas as pd
import random
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
from os import walk
from paths import glob_conf_cosyconet, glob_conf_copdgene
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-p', '--path', type=str, default='/dicom_imgs/')


if __name__ == "__main__":
    args = parser.parse_args()

    path = args.path
    save_path = glob_conf_copdgene['insp_path']

    code_array = []
    code_name_array = []
    subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
    for sub in range(0, len(subfolders)):
        for subdir, dirs, files in os.walk(subfolders[sub]):
            if 'INSP' in os.path.split(subdir)[1]:
                try:
                    directories_dicom = subdir
                    #print(directories_dicom)

                    #print(random.choice(os.listdir(directories_dicom)))
                    reader_single = sitk.ImageFileReader()
                    reader_single.SetFileName(os.path.join(directories_dicom, random.choice(os.listdir(directories_dicom))))
                    reader_single.LoadPrivateTagsOn()
                    reader_single.ReadImageInformation()
                    single_patient_id = []
                    dicts = {}
                    keys = [ '0008|0070', '0008|1090', '0018|0060', '0018|1020', '0018|1100', '0018|1210', '0010|0010']
                    for k in keys: #reader_single.GetMetaDataKeys():
                        #v = reader_single.GetMetaData(k)
                        dicts[k] = reader_single.GetMetaData(k)

                    if '' in dicts["0010|0010"]:
                        save_pickle(dicts, os.path.join(save_path, dicts["0010|0010"].replace('-', '').strip()+ ".pkl"))
                    else:
                        save_pickle(dicts, os.path.join(save_path, dicts["0010|0010"].replace('-', '') + ".pkl"))
                except:
                    print('exception', directories_dicom)

