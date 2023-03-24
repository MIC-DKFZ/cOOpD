import os
import numpy as np
import math
import SimpleITK as sitk
from batchgenerators.dataloading.data_loader import DataLoader
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.color_transforms import BrightnessTransform, BrightnessMultiplicativeTransform
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.abstract_transforms import Compose, RndTransform, AbstractTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2, MirrorTransform
from batchgenerators.transforms.crop_and_pad_transforms import RandomCropTransform
from batchgenerators.transforms.spatial_transforms import Rot90Transform
from batchgenerators.dataloading.dataset import Dataset

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
    if not (os.path.join(path, 'latent_tmp/train_cnn_latent/train_data.npz') and reconstruct is False):

        dict_train_aux = np.load(os.path.join(path, 'latent_tmp/train_cnn_latent/train_data.npz'), allow_pickle=True)
        dict_valin_aux = np.load(os.path.join(path, 'latent_tmp/train_cnn_latent/valin_data.npz'), allow_pickle=True)
        dict_train = {}
        dict_valin = {}
        for files in dict_train_aux.files:
            dict_train[files] = dict_train_aux[files]
            dict_valin[files] = dict_valin_aux[files]

        dict_output = {'Train': dict_train, 'Valin': dict_valin}

        #print(dict_output)
        reconstructed_train = []
        reconstructed_val = []
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
                if split == 'Train':
                    reconstructed_train.append(image)
                elif split == "Valin":
                    reconstructed_val.append(image)

            if split == 'Train':
                dict_output_Train = {'patients': final_patient_name, 'label': final_labels_org, 'gold': final_gold_org, 'fev': final_fev_org, 'fev_fvc': final_fev_fvc_org, 'latent': final_latent_org, 'reconstructed': reconstructed_train}
            elif split == "Valin":
                dict_output_Val = {'patients': final_patient_name, 'label': final_labels_org, 'gold': final_gold_org, 'fev': final_fev_org, 'fev_fvc': final_fev_fvc_org, 'latent': final_latent_org, 'reconstructed': reconstructed_val}

        shape_max_x = max([item.shape[1] for item in dict_output_Train['reconstructed'] + dict_output_Val['reconstructed']])
        shape_max_y = max([item.shape[2] for item in dict_output_Train['reconstructed'] + dict_output_Val['reconstructed']])
        shape_max_z = max([item.shape[3] for item in dict_output_Train['reconstructed'] + dict_output_Val['reconstructed']])
        joint_dict = [dict_output_Train, dict_output_Val]
        final_dict = []
        for rec_item in joint_dict:
            reconstructed = rec_item['reconstructed']

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
                rec_item['reconstructed'][idx_img_rec] = img_rec
                # sitk.WriteImage(sitk.GetImageFromArray(np.transpose(img_rec, (3,2,1,0))), '/home/silvia/Downloads/ex_concat.nii.gz')
                # sitk.WriteImage(sitk.GetImageFromArray(np.transpose(img_rec, (3,2,1,0))[:,:,:,140]), '/home/silvia/Downloads/ex_concat_img_' +
                #                 str(rec_item['patients'][idx_img_rec]) +
                #                 '_label_' + str(rec_item['label'][idx_img_rec])+
                #                 '_140_2_1_0.nii.gz')

            final_dict.append(rec_item)

            #dict_output_patient = {'patients': final_patient_name, 'labels': final_labels_org, 'latent': final_latent_org, 'reconstructed': reconstructed}
        np.savez_compressed(os.path.join(path, 'latent_tmp/train_cnn_latent', 'train_reconstructed'), **final_dict[0])
        np.savez_compressed(os.path.join(path, 'latent_tmp/train_cnn_latent', 'valin_reconstructed'), **final_dict[1])
        return final_dict

class COPDDataloader_cnn(DataLoader):
    def __init__(self, data, batch_size, patch_size, num_threads_in_multithreaded, seed_for_shuffle=1234, return_incomplete=False,
                 shuffle=True):
        """
        data must be a list of patients as returned by get_list_of_patients (and split by get_split_deterministic)
        patch_size is the spatial size the retured batch will have
        """
        super().__init__(data, batch_size, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete, shuffle,
                         True)
        self.patch_size = patch_size
        self.num_modalities = 512
        self.indices = list(range(len(data['labels'])))

    def generate_train_batch(self):
        # DataLoader has its own methods for selecting what patients to use next, see its Documentation
        idx = self.get_indices()
        patients_for_batch = [self._data['patients'][i] for i in idx]

        # shape = [self._data['latent'][i].shape for i in idx][0]
        # print(shape)
        # data = np.zeros((self.batch_size, *shape), dtype=np.float32)
        # print(data.shape)


        # initialize empty array for data and seg
        labels = []
        patient_names = []
        data = np.zeros((self.batch_size, self.num_modalities, *self.patch_size), dtype=np.float32)

        # iterate over patients_for_batch and include them in the batch
        for i, j in enumerate(patients_for_batch):

            patient_data, metadata = self._data['reconstructed'][self._data['patients'].index(j)], self._data['labels'][self._data['patients'].index(j)]


            data[i] = patient_data
            labels.append(metadata)
            patient_names.append(j)

        return {'data': data, 'label': labels, 'patient': patient_names}



class COPDDataloader_eval(DataLoader):
    def __init__(self, data, batch_size, patch_size, num_threads_in_multithreaded, seed_for_shuffle=1234, return_incomplete=False,
                 shuffle=True, infinite = True):
        """
        data must be a list of patients as returned by get_list_of_patients (and split by get_split_deterministic)
        patch_size is the spatial size the retured batch will have
        """
        super().__init__(data, batch_size, num_threads_in_multithreaded, shuffle, infinite, return_incomplete)
        self.patch_size = patch_size
        self.num_modalities = 512
        self.indices = list(range(len(data['label'])))

    def generate_train_batch(self):
        # DataLoader has its own methods for selecting what patients to use next, see its Documentation
        idx = self.get_indices()
        patients_for_batch = [self._data['patients'][i] for i in idx]

        # shape = [self._data['latent'][i].shape for i in idx][0]
        # print(shape)
        # data = np.zeros((self.batch_size, *shape), dtype=np.float32)
        # print(data.shape)


        # initialize empty array for data and seg
        labels = []
        gold_class = []
        fev_score = []
        fev_fvc_score = []
        patient_names = []
        data = np.zeros((self.batch_size, self.num_modalities, *self.patch_size), dtype=np.float32)

        # iterate over patients_for_batch and include them in the batch
        for i, j in enumerate(patients_for_batch):

            patient_data, label, gold, fev, fev_fvc = self._data['reconstructed'][self._data['patients'].tolist().index(j)], \
                                                      self._data['label'][self._data['patients'].tolist().index(j)], \
                                                      self._data['gold'][self._data['patients'].tolist().index(j)], \
                                                      self._data['fev'][self._data['patients'].tolist().index(j)], \
                                                      self._data['fev_fvc'][self._data['patients'].tolist().index(j)]
            #patient_data, metadata = self._data['reconstructed'][self._data['patients'].tolist().index(j)], self._data['labels'][self._data['patients'].index(j)]


            data[i] = patient_data
            labels.append(label)
            gold_class.append(gold)
            fev_score.append(fev)
            fev_fvc_score.append(fev_fvc)
            patient_names.append(j)

        return {'data': data, 'label': labels, 'patient': patient_names, 'gold': gold_class, 'fev': fev_score, 'fev_fvc': fev_fvc_score}





class COPDDataloader_unbalanced(DataLoader):
    def __init__(self, data, batch_size, patch_size, num_threads_in_multithreaded, seed_for_shuffle=1234, return_incomplete=False,
                 shuffle=True, sampling_probabilities=None, infinite = True):
        """
        data must be a list of patients as returned by get_list_of_patients (and split by get_split_deterministic)
        patch_size is the spatial size the retured batch will have
        """
        super().__init__(data, batch_size, num_threads_in_multithreaded)
        self.patch_size = patch_size
        self.num_modalities = 512
        self.indices = list(range(len(data['label'])))
        self.infinite = infinite
        self.shuffle = shuffle
        self.return_incomplete = return_incomplete
        self.seed_for_shuffle = seed_for_shuffle
        self.rs = np.random.RandomState(self.seed_for_shuffle)
        self.current_position = None
        self.was_initialized = False
        self.last_reached = False
        self.sampling_probabilities = sampling_probabilities
    def reset(self):
        assert self.indices is not None

        self.current_position = self.thread_id * self.batch_size

        self.was_initialized = True

        # no need to shuffle if we are returning infinite random samples
        if not self.infinite and self.shuffle:
            self.rs.shuffle(self.indices)

        self.last_reached = False

    def get_indices(self):
        # if self.infinite, this is easy
        if self.infinite:
            #print(self.indices)
            #print(self.sampling_probabilities)
            return np.random.choice(self.indices, self.batch_size, replace=True, p=self.sampling_probabilities)

        if self.last_reached:
            self.reset()
            raise StopIteration

        if not self.was_initialized:
            self.reset()

        indices = []

        for b in range(self.batch_size):
            if self.current_position < len(self.indices):
                indices.append(self.indices[self.current_position])

                self.current_position += 1
            else:
                self.last_reached = True
                break

        if len(indices) > 0 and ((not self.last_reached) or self.return_incomplete):
            self.current_position += (self.number_of_threads_in_multithreaded - 1) * self.batch_size
            return indices
        else:
            self.reset()
            raise

    def generate_train_batch(self):
        # DataLoader has its own methods for selecting what patients to use next, see its Documentation
        idx = self.get_indices()
        patients_for_batch = [self._data['patients'][i] for i in idx]

        # shape = [self._data['latent'][i].shape for i in idx][0]
        # print(shape)
        # data = np.zeros((self.batch_size, *shape), dtype=np.float32)
        # print(data.shape)


        # initialize empty array for data and seg
        labels = []
        gold_class = []
        fev_score = []
        fev_fvc_score = []
        patient_names = []
        data = np.zeros((self.batch_size, self.num_modalities, *self.patch_size), dtype=np.float32)

        # iterate over patients_for_batch and include them in the batch
        for i, j in enumerate(patients_for_batch):

            patient_data, label, gold, fev, fev_fvc = self._data['reconstructed'][self._data['patients'].tolist().index(j)], \
                                                      self._data['label'][self._data['patients'].tolist().index(j)], \
                                                      self._data['gold'][self._data['patients'].tolist().index(j)], \
                                                      self._data['fev'][self._data['patients'].tolist().index(j)], \
                                                      self._data['fev_fvc'][self._data['patients'].tolist().index(j)]
            #patient_data, metadata = self._data['reconstructed'][self._data['patients'].tolist().index(j)], self._data['labels'][self._data['patients'].index(j)]


            data[i] = patient_data
            labels.append(label)
            gold_class.append(gold)
            fev_score.append(fev)
            fev_fvc_score.append(fev_fvc)
            patient_names.append(j)

        return {'data': data, 'label': labels, 'patient': patient_names, 'gold': gold_class, 'fev': fev_score, 'fev_fvc': fev_fvc_score}

class ResNet_Encoder(nn.Module):
    def __init__(self, base_model="resnet50", cifar_stem=True, channels_in=4):
        """obtains the ResNet for use as an Encoder, with the last fc layer
        exchanged for an identity

        Args:
            base_model (str, optional): [description]. Defaults to "resnet50".
            cifar_stem (bool, optional): [input resolution of 32x32]. Defaults to True.
            channels_in (int, optional): [description]. Defaults to 3.
        """
        super(ResNet_Encoder, self).__init__()
        # self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
        #                     "resnet50": models.resnet50(pretrained=False)}
        self.resnet_dict = {"resnet18": ResNet18(),
                            "resnet34": ResNet34(),
                            "resnet50": ResNet50()}

        self.resnet = self._get_basemodel(base_model)
        num_ftrs = self.resnet.fc.in_features
        #num_ftrs = self.resnet.linear.in_features


        #change 1st convolution to work with inputs [channels_in x patch_x x patch_y x patch_z]
        #[channels_in x 50 x 50 x 50]
        if cifar_stem:
            conv1 = nn.Conv3d(channels_in, 64, kernel_size=3, stride=1, padding=1, bias=False)
            nn.init.kaiming_normal_(conv1.weight, mode='fan_out', nonlinearity='relu')
            self.resnet.conv1 = conv1
            self.resnet.maxpool = nn.Identity()
        elif channels_in != 3:
            conv1 = nn.Conv3d(channels_in, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
            nn.init.kaiming_normal_(conv1.weight, mode='fan_out', nonlinearity='relu')
            self.resnet.conv1 = conv1
        self.resnet.fc = nn.Identity() #should it be self.resnet.linear instead? because I changed this in 239
        if base_model == 'resnet18' or base_model == 'resnet34' or base_model == 'resnet50':
            self.z_dim = 512
        else: self.z_dim = 2048



    def forward(self, x):
        return self.resnet(x)


    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet34 or resnet50")




### ResNet following https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2, channels_in=512): #20
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv3d(channels_in, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512*block.expansion, num_classes) #self.linear
        if num_blocks[3] == 2:
            self.z_dim = 512
        else:
            self.z_dim = 2048

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # print('shape', x.shape)
        # print('shape', self.conv1(x).shape)
        # print(self.conv1)
        out = F.relu(self.bn1(self.conv1(x)))
        #print('shape', out.shape)
        out = self.layer1(out)
        #print('shape', out.shape)
        out = self.layer2(out)
        #print('shape', out.shape)
        out = self.layer3(out)
        #print('shape', out.shape)
        out = self.layer4(out)
        #print('shape', out.shape)
        #out = F.avg_pool3d(out, 4)
        #out = F.adaptive_avg_pool3d(out, (out.shape(2), out.shape(3), out.shape(4)))
        adapt = torch.nn.AdaptiveAvgPool3d(1)
        out = adapt(out)
        #print('shape', out.shape)
        out = out.view(out.size(0), -1)
        print('shape', out.shape)
        print('fc', self.fc)
        out = self.fc(out) #self.linear(out)
        print('shape', out.shape)

        return out

# def OnlyLinear(patch_size):
#     return nn.Linear(512*math.prod(patch_size), 2)

class OnlyLinear(nn.Module):
    def __init__(self, num_classes=2, channels_in=512, patch_size = (10,10,10)):
        super(OnlyLinear, self).__init__()
        self.fc = nn.Linear(channels_in * math.prod(patch_size), num_classes)

    def forward(self, x):
        #print(x.shape)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.fc(x)
        #print(x.shape)
        return x


# input: (N = batch_size, C = 512, D, H, W)
# output: (N, num_classes)
class LeNet3D(nn.Module):
    def __init__(self, num_classes=2, channels_in=512):
        super(LeNet3D, self).__init__()

        self.conv1 = nn.Conv3d(channels_in, channels_in*2, kernel_size=2)
        self.pool = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(channels_in*2, channels_in*4, kernel_size=2)
        #self.fc1 = nn.Linear(channels_in*4 * 2 * 1 * 2, 120)
        #self.fc1 = nn.Linear(channels_in*4 * 2 * 2 * 2, 120)
        self.fc1 = nn.Linear(channels_in*4 * 3 * 3 * 3, 120) #resnet18 or 34 1 channel
        #self.fc1 = nn.Linear(channels_in*4 *4 * 3 * 3, 120)  #resnet34
        #self.fc1 = nn.Linear(channels_in*4 *3 * 3 * 2, 120) #resnet18 2 channel

        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, num_classes)
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        print(x.size())
        x = self.pool(F.relu(self.conv1(x)))
        print(x.size())
        x = self.pool(F.relu(self.conv2(x)))
        print(x.size())
        x = x.view(x.size(0), -1)
        print(x.size())
        x = F.relu(self.fc1(x))
        print(x.size())
        x = self.drop(x)
        print(x.size())
        x = F.relu(self.fc2(x))
        print(x.size())
        x = self.drop(x)
        print(x.size())
        x = self.fc3(x)
        print(x.size())
        return x

class Small_LeNet(nn.Module):
    def __init__(self, num_classes=2, channels_in=512):
        super(Small_LeNet, self).__init__()
        self.pool = nn.MaxPool3d(2, 2)
        self.conv1 = nn.Conv3d(channels_in, channels_in*4, kernel_size=2)
        self.fc1 = nn.Linear(channels_in*4 * 6 * 4 * 5, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Fully_Connected(nn.Module):
    def __init__(self, num_classes=2, channels_in=512, patch_size=(10,10,10)):
        super(Fully_Connected, self).__init__()
        self.fc1 = nn.Linear(channels_in * math.prod(patch_size), 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])



def get_train_transform_cnn(mode = 'train', patch_size=(512,5,5,5), rnd_crop=False, elastic_deform=True, rotate=True, base_train='default', prob_sample = 0):
    tr_transforms = []
    if mode == "train":
        if base_train == "default":
            tr_transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.30))
        elif base_train == "gauss_bright":
            tr_transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.30))
            tr_transforms.append(BrightnessMultiplicativeTransform((0.7, 1.5), per_channel=True, p_per_sample=0.30))
        elif base_train == "spatial_transforms":
            # tr_transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.30))
            # tr_transforms.append(BrightnessMultiplicativeTransform((0.7, 1.5), per_channel=True, p_per_sample=0.30))
            tr_transforms.append(
                SpatialTransform_2(
                    patch_size, [i // 2 for i in patch_size],
                    do_elastic_deform=True, deformation_scale=(0, 0.25),
                    do_rotation=True,
                    angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
                    angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
                    angle_z=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
                    do_scale=True, scale=(0.75, 1.25),
                    border_mode_data='constant', border_cval_data=0,
                    border_mode_seg='constant', border_cval_seg=0,
                    order_seg=1, order_data=3,
                    random_crop=True,
                    p_el_per_sample=0.6, p_rot_per_sample=0.6, p_scale_per_sample=0.6
                )
            )
        elif base_train == 'best_transformations_ever':
            tr_transforms.append(GaussianBlurTransform(blur_sigma=(1, 5),
                                                       p_per_sample=prob_sample,
                                                       p_per_channel=0.5))
            # tr_transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.30))
            tr_transforms.append(MirrorTransform())
            tr_transforms.append(
                SpatialTransform_2(
                    patch_size, [min(patch_size) for _ in patch_size],
                    do_elastic_deform=True, deformation_scale=(0, 0.25),
                    do_rotation=True,
                    angle_x=(- 10 / 360. * 2 * np.pi, 10 / 360. * 2 * np.pi),
                    angle_y=(- 10 / 360. * 2 * np.pi, 10 / 360. * 2 * np.pi),
                    angle_z=(- 10 / 360. * 2 * np.pi, 10 / 360. * 2 * np.pi),
                    do_scale=True, scale=(0.9, 1.1),
                    border_mode_data='constant', border_cval_data=0,
                    border_mode_seg='constant', border_cval_seg=0,
                    order_seg=1, order_data=3,
                    random_crop=True,
                    p_el_per_sample=prob_sample, p_rot_per_sample=prob_sample, p_scale_per_sample=prob_sample
                ))
            #tr_transforms.append(Rot90Transform(p_per_sample=1./3.))

        else:
            raise NotImplementedError

        tr_transforms = Compose(tr_transforms)
    elif mode == 'offline_aug':
        tr_transforms.append(
            SpatialTransform_2(
                patch_size, [i // 2 for i in patch_size],
                do_elastic_deform=True, deformation_scale=(0, 0.25),
                do_rotation=True,
                angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
                angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
                angle_z=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
                do_scale=True, scale=(0.75, 1.25),
                border_mode_data='constant', border_cval_data=0,
                border_mode_seg='constant', border_cval_seg=0,
                order_seg=1, order_data=3,
                random_crop=True,
                p_el_per_sample=0.6, p_rot_per_sample=0.6, p_scale_per_sample=0.6
            )
        )
        tr_transforms = Compose(tr_transforms)

    return tr_transforms


def off_aug(dict, dict_aug, patch_size):
    from batchgenerators.augmentations.spatial_transformations import augment_rot90, augment_mirroring, augment_spatial_2, augment_spatial, augment_resize
    keys = list(dict.keys())
    keys.remove('reconstructed')
    for j, rec in enumerate(dict['reconstructed']):
        #aug = augment_resize(sample_data=rec, sample_seg=None, target_size=[i // 2 for i in patch_size])[0]
        #aug = augment_rot90(sample_data=rec, sample_seg=None)[0]
        #aug = augment_mirroring(sample_data= aug, sample_seg= None)[0]
        aug = np.squeeze(augment_spatial_2(data=np.expand_dims(rec, axis=0),
                                           seg=None, patch_size=[i // 1 for i in patch_size],
                                           patch_center_dist_from_border=1, deformation_scale=(0, 0.1),
                                           angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
                                           angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
                                           angle_z=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
                                           scale= (0.95, 1.05), do_scale= True, do_elastic_deform= True, do_rotation= True)[0], axis=0)
        # sitk.WriteImage(sitk.GetImageFromArray(rec[230]), '/home/silvia/Downloads/test1.nii.gz')
        # sitk.WriteImage(sitk.GetImageFromArray(aug[230]), '/home/silvia/Downloads/test2.nii.gz')


        dict_aug['reconstructed'].append(aug)
        for key in keys:
            dict_aug[key].append(dict[key][j])
    dict_aug_final = {}
    for key in dict_aug.keys():
        dict_aug_final[key] = np.asarray(dict_aug[key])

    return dict_aug_final

def activate_off_aug(dict, dict_aug, patch_size, path, name, num_aug = 2):
    dict_aug_joint = {}
    for i in range(1, num_aug+1):
        dict_aug_final = off_aug(dict, dict_aug, patch_size)
        dict_aug_joint = {key: dict_aug_final[key] for key in dict_aug_final}


    final_dict = {key: np.asarray([*dict[key], *dict_aug_joint[key]]) for key in dict}

    np.savez_compressed(os.path.join(path, 'latent_tmp', name + '_offaug_reconstructed'), **final_dict)


