
import os

glob_conf_cosyconet = dict()

glob_conf_cosyconet['insp_path'] = '/cosyconet/CT/images_data/InspT1'
glob_conf_cosyconet['reg_path'] = '/cosyconet/CT/registration/exp_to_insp_T1/final'
glob_conf_cosyconet['insp_lobes'] = '/cosyconet/CT/lung_masks/YACTA_lobe/InspT1'
glob_conf_cosyconet['lung_seg_path'] = '/cosyconet/CT/lung_masks/YACTA/InspT1'
glob_conf_cosyconet['nnUnet_trachea_aorta'] = '/nnunet_inference/cosyconet_insp_T1/nnUnet_output'
glob_conf_cosyconet['to_save'] = '/pre-processed/patches_new_all_overlap'


glob_conf_copdgene = dict()

glob_conf_copdgene['insp_path'] = '/images/Insp'
glob_conf_copdgene['reg_path'] = '/images/ExpReg/'
glob_conf_copdgene['insp_lobes'] = '/masks_nii/lung_lobe/corrected'
glob_conf_copdgene['lung_seg_path'] = '/masks_nii/lung/corrected'
glob_conf_copdgene['nnUnet_trachea_aorta'] = '/masks_nii/trachea_aorta'
glob_conf_copdgene['to_save'] = '/pre-processed/patches_new_all_overlap'
glob_conf_copdgene['patches'] = '/pre-processed/patches_new_all_overlap20/'

