import SimpleITK
import scipy
import numpy as np
import SimpleITK as sitk
import glob
import glob2
import os
import sys
from extract_patches import extract_patches_3d_fromMask, extract_allpatches_intersection, extract_allpatches_3d_fromMask, extract_allpatches_3d_fromMask_new
from paths import glob_conf_cosyconet, glob_conf_copdgene
from batchgenerators.utilities.file_and_folder_operations import *
import argparse
import torchio as tio
import random
from pathlib import Path
import time


def parse_option():
    parser = argparse.ArgumentParser('argument for pre-processing')
    # optimization
    parser.add_argument('--resample_spacing', type=float, default=[0.5, 0.5, 0.5],
                        help='new resample spacing in mm (list 3 dim)')

    parser.add_argument('--normalization', type=str, default='no_normalization',
                        choices=['no_normalization', 'aorta_trachea_norm'], help='which intensity normalization to apply')

    #CT normalization by aorta and trachea
    parser.add_argument('--erode_mask', type=float, default=[6, 6, 6],
                        help='binary erode list for trachea and aorta masks')


    print(parser.parse_args())
    opt = parser.parse_args()

    return opt




# considering new_spacing = [0.5, 0.5, 0.5] and that the optimal size for the smallest lung unit (2.5 cm)
# the smallest patch to extract should be 50 = (2.5 cm / 0.05 cm)

def resample(image, interpolator = sitk.sitkLinear, new_spacing = [0.5, 0.5, 0.5]):
    """Accepts 'image' sitk, 'interpolator' as sitk and 'new_spacing' as list of integers with length 3.
    Resamples every image to an isomorphic resolution defined by the new_spacing """""

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]
    return sitk.Resample(image, new_size, sitk.Transform(), interpolator,
                         image.GetOrigin(), new_spacing, image.GetDirection(), 0,
                         image.GetPixelID())

def corrected_CT_voxel_density(image, HU_e_air, HU_e_blood, mean_trachea, mean_aorta):
    normalized = ((HU_e_air - HU_e_blood)/(mean_trachea - mean_aorta)) * image + (HU_e_blood - mean_aorta * ((HU_e_air - HU_e_blood)/(mean_trachea - mean_aorta)))
    return normalized


def normalize_HU_nnUnet(image, segmentation_nnUNet, label):
    #based in: https://reader.elsevier.com/reader/sd/pii/S0720048X12001568?token=29B13EA2E3D8890CA0F707D35B0F03BAD667D9E170C89D9E8A78E32A797C3DD577BD0445EC9093F1794839025382D7A2&originRegion=eu-west-1&originCreation=20211118081818

    trachea_label = 3
    aorta_label = 4
    # HU expected after correction for air
    HU_e_air = -1000
    # HU expected after correction for blood
    HU_e_blood = 50

    #aorta and trachea labels
    segmentation_nnUNet.SetSpacing(image.GetSpacing())
    segmentation_nnUNet.SetDirection(image.GetDirection())
    segmentation_nnUNet.SetOrigin(image.GetOrigin())

    image_seg = sitk.BinaryErode(segmentation_nnUNet, [6, 6, 6], sitk.sitkBall, 0, trachea_label)
    image_seg = sitk.BinaryErode(image_seg, [6, 6, 6], sitk.sitkBall, 0, aorta_label)
    labelStats = sitk.LabelStatisticsImageFilter()
    labelStats.Execute(image, image_seg)
    median_trachea = labelStats.GetMedian(trachea_label)
    median_aorta = labelStats.GetMedian(aorta_label)
    mean_trachea = labelStats.GetMean(trachea_label)
    mean_aorta = labelStats.GetMean(aorta_label)
    std_trachea = labelStats.GetSigma(trachea_label)
    std_aorta = labelStats.GetSigma(aorta_label)

    #for the lung label

    label.SetSpacing(image.GetSpacing())
    label.SetDirection(image.GetDirection())
    label.SetOrigin(image.GetOrigin())
    labelStats = sitk.LabelStatisticsImageFilter()
    labelStats.Execute(image, label)
    mean_lung = labelStats.GetMean(1)
    std_lung = labelStats.GetSigma(1)
    #print('mean_lung', mean_lung, 'std', std_lung)

    LAA_950 = -950
    LAA_856 = -856
    low_limit = -1000
    #norm_LAA_950 = (LAA_950 - mean_lung)/std_lung
    norm_LAA_950 = corrected_CT_voxel_density(LAA_950, HU_e_air, HU_e_blood, mean_trachea, mean_aorta)

    #norm_LAA_856 = (LAA_856 - mean_lung)/std_lung
    norm_LAA_856 = corrected_CT_voxel_density(LAA_856, HU_e_air, HU_e_blood, mean_trachea, mean_aorta)

    #norm_low = (low_limit - mean_lung)/std_lung
    norm_low = corrected_CT_voxel_density(low_limit, HU_e_air, HU_e_blood, mean_trachea, mean_aorta)


    normalized_image_complex = corrected_CT_voxel_density(image, HU_e_air, HU_e_blood, mean_trachea, mean_aorta)
    # normalized_image_simple = (image - mean_trachea)/ (mean_aorta - mean_trachea)
    # print((mean_aorta - mean_trachea), std_lung)
    # print(median_trachea)
    # print(median_aorta)

    dict_HU = {'median_traquea': median_trachea, 'median_aorta': median_aorta,
               'mean_trachea': mean_trachea, 'mean_aorta': mean_aorta,
               'std_trachea': std_trachea, 'std_aorta': std_aorta,
               'mean_lung': mean_lung, 'std_lung': std_lung,
               'LAA_950': norm_LAA_950, 'LAA_856': norm_LAA_856,
               'norm_low': norm_low}

    return normalized_image_complex, dict_HU


def normalize_HU(image, label):
    label.SetSpacing(image.GetSpacing())
    label.SetDirection(image.GetDirection())
    label.SetOrigin(image.GetOrigin())
    labelStats = sitk.LabelStatisticsImageFilter()
    labelStats.Execute(image, label)
    mean_lung = labelStats.GetMean(1)
    std_lung = labelStats.GetSigma(1)

    #print('mean_lung', mean_lung, 'std', std_lung)

    LAA_950 = -950
    LAA_856 = -856
    low_limit = -1000
    norm_LAA_950 = (LAA_950 - mean_lung)/std_lung
    norm_LAA_856 = (LAA_856 - mean_lung)/std_lung
    norm_low = (low_limit - mean_lung)/std_lung

    #print('LAA_950', norm_LAA_950, 'LAA_856', norm_LAA_856, 'norm_low', norm_low)

    normalized_img = (image - mean_lung)/std_lung

    return normalized_img, norm_LAA_950, norm_LAA_856, norm_low

def get_lung_copd_measurements(np_insp, np_exp_reg, np_label, LAA_950, LAA_856, low_limit):
    # LAA_950 = -950
    # LAA_856 = -856
    # low_limit = -1000
    # high_limit = -810
    insp_emphysema = np.count_nonzero((np_insp < LAA_950) * (np_insp > low_limit) * np_label)
    #sitk.WriteImage(sitk.GetImageFromArray((np_insp < LAA_950) * (np_insp > low_limit) * np_label), '/home/silvia/Downloads/emph.nii.gz')
    exp_airway = np.count_nonzero((np_exp_reg < LAA_856) * (np_exp_reg > low_limit) * np_label)
    lung_count = np.count_nonzero(np_label)
    perc_emphy = insp_emphysema/lung_count
    perc_airway = exp_airway/lung_count
    mean_lung_density_insp = np.sum(np_insp * np_label)/np.sum(np_label)
    mean_lung_density_exp = np.sum(np_exp_reg * np_label)/np.sum(np_label)
    lung_perc = np.count_nonzero(np_label)/np_label.size


    return perc_emphy, perc_airway, mean_lung_density_insp, mean_lung_density_exp, lung_perc

#https://github.com/batmanlab/Subject2Vec/blob/master/src/Preprocess_Images_Create_Input_CSV.py
def Image2Patch(insp, exp, jac, labelMaskImg, labelLobeMaskImg, patchSize, finalPatchSize, acceptRate):
    """ This function converts image to patches.
        Here is the input of the function:
          inputImg : input image. This should be simpleITK object
          labelMaskImg : label image containing mask of the lung (values greater than 0)
          labelLobeMaskImg : label image containing mask of the lobes (values greater than 0)
          patchSize : size of the patch. It should be array of three scalar
          acceptRate : If portion of the patch inside of the mask exceeds value, it would be accepted
        Here is the output of the function:
          patchImgData : It is a list containing the patches of the image
          patchLblData : Is is a list containing the patches of the label image

    """
    patchVol = finalPatchSize[0] * finalPatchSize[1] * finalPatchSize[2]
    largePatchImgData = []
    largePatchImgData_exp = []
    largePatchImgData_jac = []
    largePatchImgData_label = []
    localization = []
    coordinates = []
    inputImg = insp

    #for x in range(0, inputImg.GetSize()[0] - (finalPatchSize[0] + patchSize[0]), (finalPatchSize[0] + patchSize[0])):
        #for y in range(0, inputImg.GetSize()[1] - (finalPatchSize[1] + patchSize[1]), (finalPatchSize[1] + patchSize[1])):
            #for z in range(0, inputImg.GetSize()[2] - (finalPatchSize[2] + patchSize[2]),(finalPatchSize[2] + patchSize[2])):
    for x in range(0, inputImg.GetSize()[0] - finalPatchSize[0], int((1-patchSize[0]) * finalPatchSize[0])):
        for y in range(0, inputImg.GetSize()[1] - finalPatchSize[1], int((1-patchSize[1]) * finalPatchSize[1])):
            for z in range(0, inputImg.GetSize()[2] - finalPatchSize[2], int((1-patchSize[2]) * finalPatchSize[2])):
                patchLblImg = sitk.RegionOfInterest(labelMaskImg, size=finalPatchSize, index=[x, y, z])
                npPatchLblImg = sitk.GetArrayFromImage(patchLblImg)
                if ((npPatchLblImg > 0).sum() > acceptRate * patchVol):  # if the patch has more than 70%
                    # largePatchSize = [2*patchSize[0], 2*patchSize[1], 2*patchSize[2]]
                    largePatchSize = finalPatchSize
                    # largePatchIndex = [x-patchSize[0]/2, y-patchSize[1]/2, z-patchSize[2]/2]
                    shift_x = int((finalPatchSize[0] - patchSize[0]) / 2)
                    shift_y = int((finalPatchSize[1] - patchSize[1]) / 2)
                    shift_z = int((finalPatchSize[2] - patchSize[2]) / 2)
                    largePatchIndex = [x - shift_x, y - shift_y, z - shift_z]
                    try:
                        largePatchImg = sitk.RegionOfInterest(inputImg, size=largePatchSize,
                                                              index=[x, y, z])
                        npLargePatchImg = sitk.GetArrayFromImage(largePatchImg)

                        largePatchImg_exp = sitk.RegionOfInterest(exp, size=largePatchSize,
                                                              index=[x, y, z])
                        npLargePatchImg_exp = sitk.GetArrayFromImage(largePatchImg_exp)

                        largePatchImg_jac = sitk.RegionOfInterest(jac, size=largePatchSize,
                                                              index=[x, y, z])
                        npLargePatchImg_jac = sitk.GetArrayFromImage(largePatchImg_jac)

                        largePatchImg_label = sitk.RegionOfInterest(labelMaskImg, size=largePatchSize,
                                                              index=[x, y, z])
                        npLargePatchImg_label = sitk.GetArrayFromImage(largePatchImg_label)

                        lobe_patch = sitk.RegionOfInterest(labelLobeMaskImg, size=largePatchSize,
                                                              index=[x, y, z])
                        npLargeLobePatchImg = sitk.GetArrayFromImage(lobe_patch).flatten()
                        #print(np.bincount(npLargeLobePatchImg))
                        lobe = np.argmax(np.bincount(npLargeLobePatchImg))

                        localization.append(lobe)
                        largePatchImgData.append(npLargePatchImg.copy())
                        largePatchImgData_exp.append(npLargePatchImg_exp.copy())
                        largePatchImgData_jac.append(npLargePatchImg_jac.copy())
                        largePatchImgData_label.append(npLargePatchImg_label.copy())

                        #sitk.WriteImage(sitk.GetImageFromArray(npLargePatchImg), '/home/silvia/Downloads/teste_' + str(x) + str(y) + str(z) + '.nii.gz')

                        coordinates.append([x,y,z])


                    except:
                        print("Overlapping Patch outside the largest possible region...")

    largePatchImgData = np.asarray(largePatchImgData)
    largePatchImgData_exp = np.asarray(largePatchImgData_exp)
    largePatchImgData_jac = np.asarray(largePatchImgData_jac)
    largePatchImgData_label = np.asarray(largePatchImgData_label)

    localizationData = np.asarray(localization)
    coordinatesData = np.asarray(coordinates)

    return largePatchImgData, largePatchImgData_exp, largePatchImgData_jac, largePatchImgData_label, localizationData, coordinatesData

def patch_extraction(insp, exp, jac, label, label_lobe, overlap_size, patch_size, acceptRate):
    patchImgData, patchImgData_exp, patchImgData_jac, patchImgData_label, localizationData, coordinatesData = Image2Patch(insp, exp, jac, label, label_lobe, [overlap_size, overlap_size, overlap_size], \
                               [patch_size, patch_size, patch_size], acceptRate)

    return patchImgData, patchImgData_exp, patchImgData_jac, patchImgData_label, localizationData, coordinatesData

def process_images(path_insp, path_exp_reg, path_jacobian, path_labels, path_trachea_aorta_seg, path_insp_lobe_seg,
                patch_size, overlap_size,
                  acceptRate, max_no_patches, do_aug= True):



    metadata_acq = load_pickle(path_insp.replace('images', 'metadata_CT').replace('.nii.gz', '.pkl'))


    #steps: (i) : intensity normalization of each 3D volume, using aorta and trachea mean intensities for normalization
    # (ii) re-sampling of all 3D images and corresponding labels to a fixed pixel size 0.5 using
    # linear for images and nearest-neighbour interpolation for labels


    patient_name = str(os.path.basename(path_insp)).split('.')[0]
    img_insp = sitk.ReadImage(path_insp)
    img_exp_reg = sitk.ReadImage(path_exp_reg)
    label = sitk.ReadImage(path_labels)

    label_stats = sitk.LabelStatisticsImageFilter()
    label_stats.Execute(label, label)
    label_unique = [label_stats.GetLabels()[0], label_stats.GetLabels()[1], label_stats.GetLabels()[2]]
    label_unique.sort()
    print(label_unique)

    label = sitk.BinaryThreshold(label, lowerThreshold=label_unique[1], upperThreshold=label_unique[2], insideValue=1, outsideValue=0)
    label_tr_ao = sitk.ReadImage(path_trachea_aorta_seg)
    label_lobe = sitk.ReadImage(path_insp_lobe_seg)



    try:
        img_jac = sitk.ReadImage(path_jacobian)
        img_jac.CopyInformation(img_insp)
    except ValueError: #TypeError
        img_jac = img_insp * 0


    img_exp_reg.CopyInformation(img_insp)
    label.CopyInformation(img_insp)
    label_tr_ao.CopyInformation(img_insp)
    label_lobe.CopyInformation(img_insp)


    spacing = img_insp.GetSpacing()
    direction = img_insp.GetDirection()
    origin = img_insp.GetOrigin()

    #normalize intensities: 0 mean, 1 variance
    # img_insp, norm_LAA_950, _, low_limit = normalize_HU(img_insp, label)
    # img_exp_reg, _, norm_LAA_856, _ = normalize_HU(img_exp_reg, label)
    # img_jac, _, _, _ = normalize_HU(img_jac, label)

    img_insp, dict_info = normalize_HU_nnUnet(img_insp, label_tr_ao, label)
    img_exp_reg, _ = normalize_HU_nnUnet(img_exp_reg, label_tr_ao, label)
    try:
        img_jac, _ = normalize_HU_nnUnet(img_jac, label_tr_ao, label)
    except ZeroDivisionError:
        img_jac = img_jac

    #print('LAA_950', dict_info['LAA_950'], 'LAA_856', dict_info['LAA_856'], 'norm_low', dict_info['norm_low'])
    #print('mean_trachea', dict_info['mean_trachea'], 'mean_aorta', dict_info['mean_aorta'])





    #resample to isotropic size 0.5
    img_insp = resample(img_insp)
    img_exp_reg = resample(img_exp_reg)
    img_jac = resample(img_jac)
    label = sitk.Resample(label, img_insp.GetSize(), sitk.Transform(), sitk.sitkNearestNeighbor,
                         label.GetOrigin(), img_insp.GetSpacing(), label.GetDirection(), 0,
                         label.GetPixelID())
    label_lobe = sitk.Resample(label_lobe, img_insp.GetSize(), sitk.Transform(), sitk.sitkNearestNeighbor,
                         label_lobe.GetOrigin(), img_insp.GetSpacing(), label_lobe.GetDirection(), 0,
                         label_lobe.GetPixelID())


    sitk.WriteImage(img_insp, '/home/silvia/Downloads/img_insp.nii.gz')
    sitk.WriteImage(img_exp_reg, '/home/silvia/Downloads/img_exp.nii.gz')
    #sitk.WriteImage(img_jac, '/home/silvia/Downloads/img_jac.nii.gz')
    sitk.WriteImage(label, '/home/silvia/Downloads/seg_normal.nii.gz')
    sitk.WriteImage(label_tr_ao, '/home/silvia/Downloads/seg_nnunet.nii.gz')

    patchImgData_insp, patchImgData_exp_reg, patchImgData_jac, patchImgData_label, localizationData, coordinatesData = patch_extraction(img_insp, img_exp_reg, img_jac, label, label_lobe, overlap_size, patch_size, acceptRate)

    # patchImgData_insp, localizationData, coordinatesData = patch_extraction(img_insp_aug, label_aug, label_lobe_aug, overlap_size, patch_size, acceptRate)
    # patchImgData_exp_reg, localizationData, coordinatesData = patch_extraction(img_exp_reg_aug, label_aug, label_lobe_aug, overlap_size, patch_size, acceptRate)
    # patchImgData_jac, localizationData, coordinatesData = patch_extraction(img_jac_aug, label_aug, label_lobe_aug, overlap_size, patch_size, acceptRate)
    # patchImgData_label, localizationData, coordinatesData = patch_extraction(label_aug, label_aug, label_lobe_aug, overlap_size, patch_size, acceptRate)


    for i in range(0, len(patchImgData_insp)):
        if len(patchImgData_insp) < 50:
            print('alert very small')
        patch_insp = patchImgData_insp[i]
        patch_exp = patchImgData_exp_reg[i]
        patch_jac = patchImgData_jac[i]
        patch_label = patchImgData_label[i]

        lung_emphy, lung_airway, MLD_insp, MLD_exp, lung_perc = get_lung_copd_measurements(patch_insp, patch_exp, patch_label,
                                                             dict_info['LAA_950'], dict_info['LAA_856'], dict_info['norm_low'])

        metadata_info = {
            'patient': patient_name.lower().split('_')[0],
            'patient_aug': patient_name.lower(),
            'patch_num': i,
            'spacing': spacing,
            'direction': direction,
            'origin': origin,
            'emphysema_perc': lung_emphy,
            'airway_perc': lung_airway,
            'MLD_insp': MLD_insp,
            'MLD_exp': MLD_exp,
            'lung_present': lung_perc,
            'location': str(localizationData[i]),
            'coordinates': coordinatesData[i]
        }

        #metadata = (*metadata_acq, *metadata_info)
        metadata = {**metadata_acq, **metadata_info}
        #print('lung_present',metadata['lung_present'], 'emph_perc',metadata['emphysema_perc'], 'airway_perc',metadata['airway_perc'])

        directory_save = glob_conf_copdgene['to_save'] + str(int(overlap_size*100))
        if os.path.exists(directory_save) is False:
            os.mkdir(directory_save)
        if path_jacobian == None:
            np.savez_compressed(os.path.join(directory_save,
                                             patient_name.lower() + '_' + str(i)), insp=patch_insp, exp=patch_exp,
                                jac=None, label=patch_label)
        else:
            np.savez_compressed(os.path.join(directory_save,
                                             patient_name.lower() + '_' + str(i)), insp=patch_insp, exp=patch_exp,
                                jac=patch_jac, label=patch_label)
        sitk.WriteImage(sitk.GetImageFromArray(patch_insp), '/home/silvia/Downloads/trial.nii.gz')
        sitk.WriteImage(sitk.GetImageFromArray(patch_exp), '/home/silvia/Downloads/trial_exp.nii.gz')
        sitk.WriteImage(sitk.GetImageFromArray(patch_jac), '/home/silvia/Downloads/trial_jac.nii.gz')

        save_pickle(metadata, os.path.join(directory_save,
                                           patient_name.lower() + '_' + str(i) + ".pkl"))




if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser(description='patchify data preprocessing')
    parser.add_argument('-p', '--patch_size', type=int, default=50, help='Size of patch.')
    parser.add_argument('-ol', '--overlap_size', type=float, default=0, help='% overlapping 0-1') #0.2
    parser.add_argument('-a', '--acceptRate', type=float, default=0.7,
                        help='If portion of the patch inside of the mask exceeds value, it would be accepted')
    parser.add_argument('-n', '--max_no_patches', type=int, default=1000,
                        help='The maximum number of patches in any subject.')
    parser.add_argument('-f', '--folds', type=int, default=2, help='The number of folds for cross validation.')
    parser.add_argument('-aug', '--do_aug', type=bool, default=True)
    parser.add_argument('-dt', '--dataset', type=str, choices=['cosyconet', 'copdgene'], default= 'copdgene')

    args = parser.parse_args()

    patch_size = args.patch_size
    overlap_size = args.overlap_size
    acceptRate = args.acceptRate
    max_no_patches = args.max_no_patches
    folds = args.folds
    aug = args.do_aug
    dataset = args.dataset

    #opt = parse_option()

    directories_jac = []
    directories_insp = []
    directories_exp_reg = []
    directories_label = []
    directories_nnunet = []
    directories_insp_lobe = []

    if dataset == 'cosyconet':


        for lung_insp in glob2.glob(os.path.join(glob_conf_cosyconet['insp_path'], '*.nii.gz')):
            directories_insp.append(lung_insp)
            directories_insp = sorted(directories_insp)
        print(directories_insp)
        for lung_exp_reg in glob2.glob(os.path.join(glob_conf_cosyconet['reg_path'], '*_affinebspline1.nii.gz')):
            directories_exp_reg.append(lung_exp_reg)
            directories_exp_reg = sorted(directories_exp_reg)
        print(directories_exp_reg)
        for jacobian in glob2.glob(os.path.join(glob_conf_cosyconet['reg_path'], '*_spatialjac.nii.gz')):
            directories_jac.append(jacobian)
            directories_jac = sorted(directories_jac)
        print(directories_jac)
        for insp_label in glob2.glob(os.path.join(glob_conf_cosyconet['lung_seg_path'], '*.nii.gz')):
            directories_label.append(insp_label)
            directories_label = sorted(directories_label)
        print(directories_label)
        for insp_trachea_aorta in glob2.glob(os.path.join(glob_conf_cosyconet['nnUnet_trachea_aorta'], '*.nii.gz')):
            directories_nnunet.append(insp_trachea_aorta)
            directories_nnunet = sorted(directories_nnunet)
        print(directories_nnunet)
        for lobe_insp in glob2.glob(os.path.join(glob_conf_cosyconet['insp_lobes'], '*.nii.gz')):
            directories_insp_lobe.append(lobe_insp)
            directories_insp_lobe = sorted(directories_insp_lobe)
        print(directories_insp_lobe)

        for insp in range(0, len(directories_insp)):  # len(directories_insp)
            # if os.path.basename(directories_insp[insp]) == "003258511.nii.gz":
            for label in range(0, len(directories_label)):
                if os.path.basename(directories_insp[insp]) == os.path.basename(directories_label[label]):
                    for jac in range(0, len(directories_jac)):
                        print(str(os.path.basename(directories_jac[jac])).split('_')[0] + '.nii.gz')
                        if os.path.basename(directories_insp[insp]) == \
                                str(os.path.basename(directories_jac[jac])).split('_')[0] + '.nii.gz':
                            print('igual')
                            for exp in range(0, len(directories_exp_reg)):
                                if os.path.basename(directories_insp[insp]) == \
                                        str(os.path.basename(directories_exp_reg[exp])).split('_')[0] + '.nii.gz':
                                    for nnunet_tra_aor in range(0, len(directories_nnunet)):
                                        print(os.path.basename(directories_insp[insp]),
                                              os.path.basename(directories_nnunet[nnunet_tra_aor]))
                                        if os.path.basename(directories_insp[insp]) == os.path.basename(
                                                directories_nnunet[nnunet_tra_aor]):
                                            for lobe in range(0, len(directories_insp_lobe)):
                                                print(os.path.basename(directories_insp[insp]),
                                                      os.path.basename(directories_insp_lobe[lobe]))
                                                if os.path.basename(directories_insp[insp]) == os.path.basename(
                                                        directories_insp_lobe[lobe]):
                                                    process_images(directories_insp[insp], directories_exp_reg[exp],
                                                                   directories_jac[jac], directories_label[label],
                                                                   directories_nnunet[nnunet_tra_aor],
                                                                   directories_insp_lobe[lobe], patch_size,
                                                                   overlap_size, acceptRate, max_no_patches)

    elif dataset == 'copdgene':
        for lung_insp in glob2.glob(os.path.join(glob_conf_copdgene['insp_path'], '*.nii.gz')):
            directories_insp.append(lung_insp)
            directories_insp = sorted(directories_insp)
        print(directories_insp)
        for lung_exp_reg in glob2.glob(os.path.join(glob_conf_copdgene['reg_path'], '*.nii.gz')):
            directories_exp_reg.append(lung_exp_reg)
            directories_exp_reg = sorted(directories_exp_reg)
        print(directories_exp_reg)

        for insp_label in glob2.glob(os.path.join(glob_conf_copdgene['lung_seg_path'], '*.nii.gz')):
            directories_label.append(insp_label)
            directories_label = sorted(directories_label)
        print(directories_label)
        for insp_trachea_aorta in glob2.glob(os.path.join(glob_conf_copdgene['nnUnet_trachea_aorta'], '*.nii.gz')):
            directories_nnunet.append(insp_trachea_aorta)
            directories_nnunet = sorted(directories_nnunet)
        print(directories_nnunet)
        for lobe_insp in glob2.glob(os.path.join(glob_conf_copdgene['insp_lobes'], '*.nii.gz')):
            directories_insp_lobe.append(lobe_insp)
            directories_insp_lobe = sorted(directories_insp_lobe)
        print(directories_insp_lobe)

        # files_exist = os.listdir(os.path.join(glob_conf_copdgene['to_save'] + str(int(overlap_size*100))))
        # files_exist = list(set([x.split('_')[0] + '_' + x.split('_')[1] for x in files_exist]))

        for insp in range(0, len(directories_insp)):  # len(directories_insp)
            # if os.path.basename(directories_insp[insp]) == "003258511.nii.gz":
            for label in range(0, len(directories_label)):
                if os.path.basename(directories_insp[insp]) == os.path.basename(directories_label[label]):
                    for exp in range(0, len(directories_exp_reg)):
                        if os.path.basename(directories_insp[insp]) == str(os.path.basename(directories_exp_reg[exp])):
                            for nnunet_tra_aor in range(0, len(directories_nnunet)):
                                print(os.path.basename(directories_insp[insp]),os.path.basename(directories_nnunet[nnunet_tra_aor]))
                                if os.path.basename(directories_insp[insp]) == os.path.basename(directories_nnunet[nnunet_tra_aor]):
                                    for lobe in range(0, len(directories_insp_lobe)):
                                        print(os.path.basename(directories_insp[insp]),os.path.basename(directories_insp_lobe[lobe]))
                                        if os.path.basename(directories_insp[insp]) == os.path.basename(directories_insp_lobe[lobe]):
                                            #if str(os.path.basename(directories_insp[insp])).split('.')[0].lower() not in files_exist:
                                            #if not glob.glob(os.path.join(glob_conf_copdgene['to_save'] + str(int(overlap_size*100)), str(os.path.basename(directories_insp[insp])).split('.')[0].lower()) + '*.npz'):
                                            #if time.time() - os.path.getmtime(directories_exp_reg[exp]) < (10000):
                                            print(insp)
                                            process_images(directories_insp[insp], directories_exp_reg[exp],
                                                           None, directories_label[label],
                                                           directories_nnunet[nnunet_tra_aor],
                                                           directories_insp_lobe[lobe], patch_size,
                                                           overlap_size, acceptRate, max_no_patches)
                                            # else:
                                            #     pass



    else:
        exit()








