import os
from argparse import ArgumentParser

import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

from pyutils.parser import str2bool
from latent_gen.ood_model import Abstract_OOD
from datamodules.brain_module import BrainDataModule
from algo.model import load_best_model, get_label_latent_forCNN
from config.latent_model import filename, model_dicts, tmp, suffix, rel_save
import torch
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from latent_gen.attention_mech import Attention, COPDDataloader, reconstruct_vector_attentionmech, get_train_transform, MixOrderTransform, COPDDataloader_attmech_unbalanced
import torch.optim as optim
from torch.autograd import Variable
import shutil
from torch.utils.tensorboard import SummaryWriter
import SimpleITK as sitk
from latent_gen.cnn_aftercradl import COPDDataloader_eval, off_aug, activate_off_aug, COPDDataloader_unbalanced, ResNet_Encoder, ResNet18, ResNet50, get_train_transform_cnn, OnlyLinear, LeNet3D, Small_LeNet, Fully_Connected
from models import base
import torch.nn as nn
from data_aug.bg_wrapper import get_simclr_pipeline_transform
from collections import Counter
import pickle
from sklearn.metrics import roc_auc_score, recall_score, precision_score, precision_recall_curve, auc
import yaml
import math
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score
import pandas as pd

parser = ArgumentParser()
parser.add_argument('-p', '--path', type=str, default='/home/silvia/Documents/CRADL/logs_cradl/copdgene/pretext/brain/simclr-resnet34/default/17674151') #12085919')#simclr-VGG13/default/10920176')#simclr-VGG16/default/11007765')
parser.add_argument('--num_epoch', type=int, default=1)
parser.add_argument('--resave', type=str2bool, nargs='?', const=False, default=False)
parser.add_argument('--reconstruct', type=str2bool, nargs='?', const=False, default=False)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--wd', type=float, default=3e-5)
parser.add_argument("--model_type", choices=['resnet18', 'resnet50', 'linear', 'LeNet3D', 'Small_LeNet', 'Fully_Connected'], default='LeNet3D', type=str)
parser.add_argument("--classification_type", choices=['binary', 'multiclass'], default='binary', type=str)
parser.add_argument("--input", default='insp_exp_reg', type=str,
                    choices=['insp', 'insp_exp_reg', 'insp_jacobian', 'jacobian'])
parser.add_argument("--overlap", default='20', type=str, choices=['0', '20'])
parser.add_argument("--realworld_dataset", default=False)
parser.add_argument('-exp', '--experiment', type=str, default='version_16')


def save_outputs(path, resave=False, num_epoch=1, get_slice_idx=True, input='insp', overlap = '0', realworld_dataset= False):
    tmp_dir = os.path.join(path, tmp, 'train_cnn_latent')
    suffix = '_data'
    keys = ['Test']

    #print(all([os.path.exists(os.path.join(tmp_dir, key.lower()+suffix+'.npz')) for key in keys]))
    if not (all([os.path.exists(os.path.join(tmp_dir, key.lower()+suffix+'.npz')) for key in keys]) and resave is False):
        experiment, args = load_best_model(path)
        experiment = experiment.to('cuda:0')
        mode='fit'
        if num_epoch != 1:
            mode = 'train'
        datamodule = BrainDataModule(mode=mode, batch_size=64, step = 'test', input=input, overlap = overlap, realworld_dataset=realworld_dataset) #attention_mech #fitting_GMM

        #loader_dict = {'train': datamodule.train_dataloader(), 'val':datamodule.val_dataloader()}
        test_loader = datamodule.test_dataloader()
        loader_dict = {'test': test_loader}

        data_dict = dict()
        # from pdb import set_trace as bp
        # bp()

        data_dict['test']= get_label_latent_forCNN(experiment, loader_dict['test'], get_slice=get_slice_idx, dir= tmp_dir)

        if os.path.exists(tmp_dir) is False:
            os.mkdir(tmp_dir)
        for key in data_dict.keys():
            np.savez_compressed(os.path.join(tmp_dir, key.lower()+suffix), **data_dict[key])
        return data_dict

def helper_coord_old(first_coord, sec_coord, patch_size):
    transformed = list((np.array(sec_coord) - np.array(first_coord)) % patch_size)
    return transformed

def helper_coord(first_coord, sec_coord, patch_size):
    transformed = list((np.array(sec_coord) - np.array(first_coord)) / patch_size)
    transformed = [math.ceil(x) for x in transformed]
    return transformed
def helper_coord_overlap(first_coord, sec_coord, patch_size, overlap):
    overlap = int(overlap)/100
    transformed = list(((np.array(sec_coord) - np.array(first_coord)) % patch_size) * (1-overlap))
    transformed = [int(x) for x in transformed]
    return transformed
def reconstruct_img(path, reconstruct = False):
    if not (os.path.join(path, 'latent_tmp/train_cnn_latent/test_data.npz') and reconstruct is False):

        dict_test_aux = np.load(os.path.join(path, 'latent_tmp/train_cnn_latent/test_data.npz'), allow_pickle=True)
        dict_test = {}
        for files in dict_test_aux.files:
            dict_test[files] = dict_test_aux[files]

        dict_output = {'Test': dict_test}

        print('doing reconstruct')
        reconstructed_test = []
        for split in dict_output:
            patient_name = np.hstack(dict_output[split]['patient'])
            patch_num = dict_output[split]['patch_number']
            location = dict_output[split]['location']
            coordinates = dict_output[split]['coordinates']
            latent = dict_output[split]['latent']
            gold = dict_output[split]['gold']
            fev = dict_output[split]['fev']
            fev_fvc = dict_output[split]['fev_fvc']
            labels = dict_output[split]['labels'] #.cpu().detach().numpy()

            p = patient_name.argsort()

            patient_name_org = patient_name[p]
            patch_num_org = patch_num[p]
            location_org = location[p]
            coordinates_org = coordinates[p]
            latent_org = latent[p]
            labels_org = labels[p]
            gold_org = gold[p]
            fev_org = fev[p]
            fev_fvc_org = fev_fvc[p]
            indexes = [index for index, _ in enumerate(patient_name_org) if
                       patient_name_org[index] != patient_name_org[index - 1]]
            indexes.append(len(patient_name_org))
            final_patient_name = [patient_name_org[indexes[i]:indexes[i + 1]] for i, _ in enumerate(indexes) if
                                  i != len(indexes) - 1]
            final_patch_num_org = [patch_num_org[indexes[i]:indexes[i + 1]] for i, _ in enumerate(indexes) if
                                   i != len(indexes) - 1]
            final_location_org = [location_org[indexes[i]:indexes[i + 1]] for i, _ in enumerate(indexes) if
                                  i != len(indexes) - 1]
            final_coordinates_org = [coordinates_org[indexes[i]:indexes[i + 1]] for i, _ in enumerate(indexes) if
                                  i != len(indexes) - 1]
            final_latent_org = [latent_org[indexes[i]:indexes[i + 1]] for i, _ in enumerate(indexes) if
                                i != len(indexes) - 1]
            final_labels_org = [labels_org[indexes[i]:indexes[i + 1]] for i, _ in enumerate(indexes) if
                                i != len(indexes) - 1]
            final_gold_org = [gold_org[indexes[i]:indexes[i + 1]] for i, _ in enumerate(indexes) if
                                i != len(indexes) - 1]
            final_fev_org = [fev_org[indexes[i]:indexes[i + 1]] for i, _ in enumerate(indexes) if
                                i != len(indexes) - 1]
            final_fev_fvc_org = [fev_fvc_org[indexes[i]:indexes[i + 1]] for i, _ in enumerate(indexes) if
                                i != len(indexes) - 1]
            final_patient_name = [max(map(str, i)) for i in final_patient_name]
            final_labels_org = [max(map(int, i)) for i in final_labels_org]
            final_gold_org = [int(i[0]) if not math.isnan(i[0]) else math.nan for i in final_gold_org]
            final_fev_org = [int(i[0]) if not math.isnan(i[0]) else math.nan for i in final_fev_org]
            final_fev_fvc_org = [int(i[0]) if not math.isnan(i[0]) else math.nan for i in final_fev_fvc_org]

            # final_gold_org = [max(map(int, i)) for i in final_gold_org]
            # final_fev_org = [max(map(int, i)) for i in final_fev_org]
            # final_fev_fvc_org = [max(map(float, i)) for i in final_fev_fvc_org]




            for patient in range(0, len(final_coordinates_org)):
                coordinates_sorted = sorted(final_coordinates_org[patient], key=lambda k: [k[0], k[1], k[2]])
                #print(coordinates_sorted)
                patch_size = 50
                shape_img_x = max([item[0] for item in coordinates_sorted])
                shape_img_y = max([item[1] for item in coordinates_sorted])
                shape_img_z = max([item[2] for item in coordinates_sorted])
                shape_base_img_x = min([item[0] for item in coordinates_sorted])
                shape_base_img_y = min([item[1] for item in coordinates_sorted])
                shape_base_img_z = min([item[2] for item in coordinates_sorted])
                shape_img = helper_coord([shape_base_img_x, shape_base_img_y, shape_base_img_z], [shape_img_x, shape_img_y, shape_img_z], patch_size)
                #image = np.zeros((shape_img[0] + 1, shape_img[1] + 1, (shape_img[2]+1)*final_latent_org[patient].shape[1]))
                image = np.zeros((final_latent_org[patient].shape[1], shape_img[0] + 1, shape_img[1] + 1, shape_img[2] + 1))

                for index in coordinates_sorted:
                    current_coord = helper_coord([shape_base_img_x, shape_base_img_y, shape_base_img_z], index, patch_size)
                    # if current_coord[2] == 0:
                    #     print(current_coord[0], current_coord[1], 0)
                    #     print(image[current_coord[0]][current_coord[1]][0])
                    #     image[current_coord[0]][current_coord[1]][0:final_latent_org[patient].shape[1]] = final_latent_org[patient][np.argmax(np.bincount([np.where(final_coordinates_org[0] == index)][0][0]))]
                    # else:
                    #print(int(final_latent_org[patient].shape[1] * current_coord[2]))
                    #print(int(final_latent_org[patient].shape[1] * current_coord[2]) + final_latent_org[patient].shape[1])
                    #print(index)
                    #print(final_latent_org[patient][np.argmax(np.bincount(np.where(final_coordinates_org[patient] == index)[0]))])
                    #image[current_coord[0]][current_coord[1]][int(final_latent_org[patient].shape[1] * current_coord[2]): int(final_latent_org[patient].shape[1] * current_coord[2]) + final_latent_org[patient].shape[1]] = final_latent_org[patient][np.argmax(np.bincount([np.where(final_coordinates_org[0] == index)][0][0]))]
                    #image[current_coord[0]][current_coord[1]][int(final_latent_org[patient].shape[1] * current_coord[2]): int(final_latent_org[patient].shape[1] * current_coord[2]) + final_latent_org[patient].shape[1]] = final_latent_org[patient][np.argmax(np.bincount(np.where(final_coordinates_org[patient] == index)[0]))]

                    for value in range(0, len(final_latent_org[patient][np.argmax(np.bincount(np.where(final_coordinates_org[patient] == index)[0]))])):
                        image[value][current_coord[0]][current_coord[1]][current_coord[2]] = final_latent_org[patient][np.argmax(np.bincount(np.where(final_coordinates_org[patient] == index)[0]))][value]
                    #print(current_coord)
                    #print(index)
                reconstructed_test.append(image)


            dict_output_Test = {'patients': final_patient_name, 'label': final_labels_org, 'gold': final_gold_org, 'fev': final_fev_org, 'fev_fvc': final_fev_fvc_org, 'latent': final_latent_org, 'reconstructed': reconstructed_test}
        # size normal 17, 16, 19
        # size real world 17, 12, 16
        shape_max_x = 17 # max([item.shape[1] for item in dict_output_Test['reconstructed']])
        shape_max_y = 12 # max([item.shape[2] for item in dict_output_Test['reconstructed']])
        shape_max_z = 16 # max([item.shape[3] for item in dict_output_Test['reconstructed']])
        final_dict = []
        reconstructed = dict_output_Test['reconstructed']

        for idx_img_rec in range(0, len(reconstructed)):
            # sitk.WriteImage(sitk.GetImageFromArray(np.transpose(reconstructed[idx_img_rec], (3,2,1,0))), '/home/silvia/Downloads/ex_original_3_2_1_0.nii.gz')
            # sitk.WriteImage(sitk.GetImageFromArray(np.transpose(reconstructed[idx_img_rec], (3,2,1,0))[:,:,:,0]), '/home/silvia/Downloads/ex_original_img1_2_1_0.nii.gz')
            # sitk.WriteImage(sitk.GetImageFromArray(np.transpose(reconstructed[idx_img_rec], (3,2,1,0))[:,:,:,10]), '/home/silvia/Downloads/ex_original_img10_2_1_0.nii.gz')
            # sitk.WriteImage(sitk.GetImageFromArray(np.transpose(reconstructed[idx_img_rec], (3,2,1,0))[:,:,:,60]), '/home/silvia/Downloads/ex_original_img60_2_1_0.nii.gz')
            # sitk.WriteImage(sitk.GetImageFromArray(np.transpose(reconstructed[idx_img_rec], (3,2,1,0))[:,:,:,100]), '/home/silvia/Downloads/ex_original_img100_2_1_0.nii.gz')
            # sitk.WriteImage(sitk.GetImageFromArray(np.transpose(reconstructed[idx_img_rec], (3,2,1,0))[:,:,:,140]), '/home/silvia/Downloads/ex_original_img140_2_1_0.nii.gz')
            #

            img_rec = np.concatenate([reconstructed[idx_img_rec], np.zeros((reconstructed[idx_img_rec].shape[0], shape_max_x - reconstructed[idx_img_rec].shape[1], reconstructed[idx_img_rec].shape[2], reconstructed[idx_img_rec].shape[3]))], axis=1)
            img_rec = np.concatenate([img_rec, np.zeros((img_rec.shape[0], img_rec.shape[1], shape_max_y - img_rec.shape[2], img_rec.shape[3]))], axis=2)
            img_rec = np.concatenate([img_rec, np.zeros((img_rec.shape[0], img_rec.shape[1], img_rec.shape[2], shape_max_z - img_rec.shape[3]))], axis=3)
            dict_output_Test['reconstructed'][idx_img_rec] = img_rec
            # sitk.WriteImage(sitk.GetImageFromArray(np.transpose(img_rec, (3,2,1,0))), '/home/silvia/Downloads/ex_concat.nii.gz')
            # sitk.WriteImage(sitk.GetImageFromArray(np.transpose(img_rec, (3,2,1,0))[:,:,:,140]), '/home/silvia/Downloads/ex_concat_img_' +
            #                 str(rec_item['patients'][idx_img_rec]) +
            #                 '_label_' + str(rec_item['label'][idx_img_rec])+
            #                 '_140_2_1_0.nii.gz')

        final_dict.append(dict_output_Test)

        np.savez_compressed(os.path.join(path, 'latent_tmp/train_cnn_latent', 'test_reconstructed'), **final_dict[0])
        return final_dict


def evaluation(n_epochs, test_gen, model, criterion, label_name, device, experience_path, eval:bool):
    if eval:
        valid_loss = 0.
        total = 0
        correct = 0
        prediction = []
        true_value = []
        patient_names = []

        for b in range(n_epochs):
            batch = next(test_gen)
            print('Start Testing')
            with torch.no_grad():
                model.eval()
                data = torch.from_numpy(batch['data']).float().to(device)
                print('label',label_name)
                labels = torch.from_numpy(np.array(batch[label_name])).long().to(device)
                outputs = model(data)
                #print('outputs', outputs)

                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                _, predicted = outputs.max(1)

                total += labels.size(0)
                print('labels_size', labels.size(0))
                correct += predicted.eq(labels).sum().item()

                prediction.append(predicted.cpu().detach().numpy())
                true_value.append(labels.cpu().detach().numpy())


                n = batch['patient']
                label = labels[0].cpu().detach().numpy()
                #print(n, label, correct)
                print('prediction', str(n[0]) + "\t" + str(label) + "\t" + str(int(correct)))
                print(labels.cpu().detach().numpy())
                print(predicted.cpu().detach().numpy())
                patient_names.append(n)

                print(' Test Set, Batch_index: {:.0f}, Loss: {:.4f}, Acc: {:.4f}, Correct: {:.0f}, Total: {:.0f}'.format(b, valid_loss/(b+1), 100.*correct/total, correct, total))

        # calculate average losses
        print('total',total)
        valid_acc = 100.*correct/total
        # print(valid_acc)
        # print(true_value)
        # print(prediction)
        print(patient_names)
        predictions_flatten = [item for sublist in prediction for item in sublist]
        true_value_flatten = [item for sublist in true_value for item in sublist]
        patient_names_flatten = [item for sublist in patient_names for item in sublist]
        print(len(predictions_flatten), predictions_flatten)
        print(len(true_value_flatten), true_value_flatten)
        print(len(patient_names_flatten), patient_names_flatten)

        # New lists
        patient_names_flatten_clean = []
        predictions_flatten_clean = []
        true_value_flatten_clean = []
        print(len(patient_names_flatten))
        print(len(predictions_flatten))
        print(len(true_value_flatten))
        # # Loop through elements of input lists
        # for a, b, c in zip(patient_names_flatten, predictions_flatten, true_value_flatten):
        #     # If it isn't a duplicate, append to the new lists
        #     if a not in patient_names_flatten_clean:
        #         patient_names_flatten_clean.append(a)
        #         predictions_flatten_clean.append(b)
        #         true_value_flatten_clean.append(c)
        #
        # # Output new lists
        # print(len(patient_names_flatten_clean))
        # print(len(predictions_flatten_clean))
        # print(len(true_value_flatten_clean))

        df_predictions = pd.DataFrame({'patient': patient_names_flatten, 'predictions_softmax': predictions_flatten})

        test_dict = {}
        test_dict['acc'] = float(accuracy_score(true_value_flatten, predictions_flatten))
        test_dict['auroc'] = float(roc_auc_score(true_value_flatten, predictions_flatten))
        # Data to plot precision - recall curve
        precision, recall, thresholds = precision_recall_curve(true_value_flatten, predictions_flatten)
        # Use AUC function to calculate the area under the curve of precision recall curve
        test_dict['auprc'] = float(auc(recall, precision))
        print(test_dict)
        print(experience_path)
        with open(os.path.join(experience_path, 'predictions_test_results.yaml'), 'w+') as file:
            yaml.dump(test_dict, file)
        df_predictions.to_csv(os.path.join(experience_path, 'predictions_full_test_results.csv'))





if __name__ == "__main__":
    args = parser.parse_args()
    path = args.path
    resave =  args.resave
    reconstruct =  args.reconstruct
    num_epoch = args.num_epoch
    batch_size = args.batch_size
    weight_decay = args.wd
    model_type = args.model_type
    classification_type = args.classification_type
    input = args.input
    overlap = args.overlap
    realworld_dataset = args.realworld_dataset
    path_to_experiment = os.path.join(path, 'results_plot_0/cnn/runs/label', args.experiment)


    if classification_type == 'binary':
        num_classes = 2
        label_name = 'label'
    elif classification_type == 'multiclass':
        num_classes = 5
        label_name = 'gold'
    else:
        NotImplemented


    save_outputs(path, get_slice_idx=False, resave=resave, num_epoch=num_epoch, input=input, overlap = overlap, realworld_dataset=realworld_dataset)

    reconstruct_img(path, reconstruct=reconstruct)

    test_aux = np.load(os.path.join(path, 'latent_tmp/train_cnn_latent', 'test_reconstructed.npz'), allow_pickle=True)
    test = {}
    for files in test_aux.files:
        test[files] = test_aux[files]

    num_channels = test['latent'][0].shape[1]
    patch_size = test['reconstructed'][0][0].shape

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LeNet3D(num_classes=num_classes, channels_in=num_channels).to(device)
    ch = torch.load(os.path.join(path_to_experiment, 'best_model', 'best_model.pt'))
    model.load_state_dict(ch['state_dict'])
    model.cuda()

    criterion = nn.CrossEntropyLoss()

    dataloader_test = COPDDataloader_eval(test, batch_size, patch_size, 1, shuffle=False) #prob_train

    # test_gen = MultiThreadedAugmenter(dataloader_test_unbalanced, None,
    #                                  num_processes=max(1, 8 // 2), num_cached_per_queue=1,
    #                                  seeds=None,
    #                                  pin_memory=False)

    test_run = evaluation(n_epochs=int(len(test['reconstructed'])/batch_size), test_gen= dataloader_test, model= model, criterion= criterion, label_name= label_name, device=device, experience_path= path_to_experiment, eval=True)



    #evaluation(1, num_epochs, num_batches_per_epoch, num_val_batches_per_epoch, np.Inf, 0, model, tr_gen, val_gen, device, optimizer, criterion,
                          #"current_checkpoint.pt", "best_model.pt", scheduler, label_name)



