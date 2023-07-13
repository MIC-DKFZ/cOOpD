import os
import numpy as np
import SimpleITK as sitk
import sys
import glob2
import pandas as pd
from parametermap_config import parameter_file
from paths import glob_conf_cosyconet, glob_conf_copdgene


print(sitk.Version())
def register(fixed, moving, save_path):
    af, b1, b2, p_ab1, p_ab1b2 = parameter_file()

    fixed_img = sitk.ReadImage(fixed)
    moving_img = sitk.ReadImage(moving)

    #  affine and b-spline_1
    # no masks
    elastixImageFilter_2 = sitk.ElastixImageFilter()
    elastixImageFilter_2.SetFixedImage(fixed_img)
    elastixImageFilter_2.SetMovingImage(moving_img)
    elastixImageFilter_2.SetParameterMap(p_ab1)
    elastixImageFilter_2.LogToConsoleOn()
    sitk.PrintParameterMap(p_ab1)
    elastixImageFilter_2.Execute()
    sitk.WriteImage(sitk.Cast(elastixImageFilter_2.GetResultImage(), sitk.sitkInt16), os.path.join(save_path, 'new_reg', os.path.basename(moving)))



if __name__ == '__main__':
    directories_exp = []
    directories_insp = []
    directories_ExpReg = []
    directories_mask_insp = []

    path_insp = glob_conf_cosyconet['insp_path']
    path_exp = glob_conf_copdgene['exp_path']
    path_ExptoInsp_reg = glob_conf_copdgene['reg_path']
    path_lung = glob_conf_copdgene['insp_lobes']
    path_lobe = glob_conf_copdgene['lung_seg_path']

    for lung_exp in glob2.glob(os.path.join(path_exp, '*.nii.gz')):
        directories_exp.append(lung_exp)
        directories_exp = sorted(directories_exp)
    print(directories_exp)

    for lung_insp in glob2.glob(os.path.join(path_insp, '*.nii.gz')):
        directories_insp.append(lung_insp)
        directories_insp = sorted(directories_insp)
    print(directories_insp)

    for mask_insp in glob2.glob(os.path.join(path_lung, '*.nii.gz')):
        directories_mask_insp.append(mask_insp)
        directories_mask_insp = sorted(directories_mask_insp)
    print(directories_mask_insp)

    for ExpReg in glob2.glob(os.path.join(path_ExptoInsp_reg, '*.nii.gz')):
        directories_ExpReg.append(ExpReg)
        directories_ExpReg = sorted(directories_ExpReg)
    print(directories_ExpReg)

    for exp in range(0, len(directories_exp)):
        for insp in range(0, len(directories_insp)):
            if os.path.basename(directories_exp[exp]) == os.path.basename(directories_insp[insp]) and not (os.path.isfile(os.path.join(path_ExptoInsp_reg, os.path.basename(directories_exp[exp]))) or os.path.isfile(os.path.join(path_ExptoInsp_reg, 'new_reg', os.path.basename(directories_exp[exp])))):
                print(os.path.basename(directories_exp[exp]))
                print(os.path.basename(directories_insp[insp]))
                register(fixed=directories_insp[insp], moving=directories_exp[exp], save_path=path_ExptoInsp_reg)
