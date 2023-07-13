import os 
from argparse import ArgumentParser

import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

from pyutils.parser import str2bool
from latent_gen.ood_model import Abstract_OOD
from datamodules.lung_module import LungDataModule
from algo.model import load_best_model, get_label_latent_forCNN
from config.latent_model import filename, model_dicts, tmp, suffix, rel_save
import torch
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from latent_gen.attention_mech import Attention, COPDDataloader, reconstruct_vector_attentionmech, get_train_transform, MixOrderTransform, COPDDataloader_attmech_unbalanced
import torch.optim as optim
from torch.autograd import Variable
import shutil
from torch.utils.tensorboard import SummaryWriter
#import SimpleITK as sitk
from latent_gen.cnn_aftercradl import reconstruct_img, COPDDataloader_eval, off_aug, activate_off_aug, COPDDataloader_unbalanced, ResNet_Encoder, ResNet18, ResNet50, ResNet34, get_train_transform_cnn, OnlyLinear, LeNet3D, Small_LeNet, Fully_Connected
from models import base
import torch.nn as nn
from data_aug.bg_wrapper import get_simclr_pipeline_transform
from collections import Counter
import pickle
from sklearn.metrics import roc_auc_score, recall_score, precision_score, precision_recall_curve, auc
import yaml



parser = ArgumentParser()
parser.add_argument('-p', '--path', type=str, default='/home/silvia/Documents/CRADL/logs_cradl/copdgene/pretext/lung/simclr-resnet34/default/17142285') #12085919')#simclr-VGG13/default/10920176')#simclr-VGG16/default/11007765')
parser.add_argument('--num_epoch', type=int, default=1)
parser.add_argument('--resave', type=str2bool, nargs='?', const=False, default=False)
parser.add_argument('--reconstruct', type=str2bool, nargs='?', const=False, default=False)
parser.add_argument('--base_train', type=str, default='best_transformations_ever')
parser.add_argument('--batch_size', type=int, default=40) #40
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--wd', type=float, default=3e-5)
parser.add_argument('--probs', type=float, default=0.2)
parser.add_argument("--model_type", choices=['resnet18', 'resnet50', 'resnet34', 'linear', 'LeNet3D', 'Small_LeNet', 'Fully_Connected'], default='LeNet3D', type=str)
parser.add_argument("--exp_type", choices=['attention', 'cnn'], default='cnn', type=str)
parser.add_argument("--classification_type", choices=['binary', 'multiclass'], default='binary', type=str)
parser.add_argument("--input", default='insp', type=str,
                    choices=['insp', 'insp_exp_reg', 'insp_jacobian', 'jacobian'])
parser.add_argument("--overlap", default='20', type=str, choices=['0', '20'])
parser.add_argument("--realworld_dataset", default=False)


def fit_model(model:Abstract_OOD, X, Y, save_path, mode='Train'):
    model.fit(X, Y)
    model.setup(X, Y, mode=mode)
    model.save_model(save_path, filename=filename)
    return model

def init_model(model_name, path, rel_save=None):
    if rel_save is not None:
        rel_save = os.path.join(path, rel_save)
        if not os.path.exists(rel_save):
            os.makedirs(rel_save)
    save_path = os.path.join(rel_save, model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return save_path

def save_outputs(path, resave=False, num_epoch=1, get_slice_idx=True, input='insp', overlap = '0', realworld_dataset= False):
    tmp_dir = os.path.join(path, tmp, 'train_cnn_latent')
    suffix = '_data'
    keys = ['Train', 'Valin']

    #print(all([os.path.exists(os.path.join(tmp_dir, key.lower()+suffix+'.npz')) for key in keys]))
    if not (all([os.path.exists(os.path.join(tmp_dir, key.lower()+suffix+'.npz')) for key in keys]) and resave is False):
        experiment, args = load_best_model(path)
        experiment = experiment.to('cuda:0')
        mode='fit'
        if num_epoch != 1:
            mode = 'train'
        datamodule = LungDataModule(mode=mode, batch_size=64, step = 'train_cnn_latent', input=input, overlap = overlap, realworld_dataset=realworld_dataset) #attention_mech #fitting_GMM

        #loader_dict = {'train': datamodule.train_dataloader(), 'val':datamodule.val_dataloader()}
        train_loader, val_loader = datamodule.train_dataloader()
        loader_dict = {'train': train_loader, 'val': val_loader}

        data_dict = dict()
        # from pdb import set_trace as bp 
        # bp()
        for key1, key2 in zip(keys, ['train', 'val']):
            if key1 =='Train':
                data_dict[key1]= get_label_latent_forCNN(experiment, loader_dict[key2], get_slice=get_slice_idx, num_epoch=num_epoch, dir= tmp_dir)
            else:
                data_dict[key1]= get_label_latent_forCNN(experiment, loader_dict[key2], get_slice=get_slice_idx, dir= tmp_dir)

        if os.path.exists(tmp_dir) is False:
            os.mkdir(tmp_dir)
        for key in data_dict.keys():
            np.savez_compressed(os.path.join(tmp_dir, key.lower()+suffix), **data_dict[key])
        return data_dict
            
def load_tmp_data(path, mode='val', get_slice=False):
    loading_path = os.path.join(path, tmp, mode+suffix+'.npz')
    loaded = np.load(loading_path)
    
    if get_slice:
        return loaded["latent"], loaded['labels'], loaded['slice_idxs']
    return loaded["latent"], loaded['labels']

def load_outputs(path, c=0):
    X, Y = dict(), dict()
    X['Train'], Y['Train'] = load_tmp_data(path, mode='train')
    X['Valin'], Y['Valin'] = load_tmp_data(path, mode='valin')
    if c is not None:
        for k in Y.keys():
            Y[k] = get_ood_labels(Y[k],c=c)
    return X, Y


def get_ood_labels(y, c=0):
    mask = y==c
    y[mask] = 0
    y[~mask] = 1
    return y





def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)

def train_model_attmech(start_epochs, n_epochs, num_batches_per_epoch, num_val_batches_per_epoch, valid_loss_min_input, model, tr_gen, val_gen, device, optimizer, checkpoint_path,
          best_model_path, scheduler):
    """
    Keyword arguments:
    start_epochs -- the real part (default 0.0)
    n_epochs -- the imaginary part (default 0.0)
    valid_loss_min_input
    loaders
    model
    optimizer
    criterion
    use_cuda
    checkpoint_path
    best_model_path

    returns trained model
    """
    # initialize tracker for minimum validation loss
    valid_loss_min = valid_loss_min_input

    save_path = init_model('attention_mech', path, rel_save=rel_save)
    print('save_path', save_path)
    print(checkpoint_path)
    print('full_path', os.path.join(save_path, 'checkpoint', checkpoint_path))

    #create folder
    if not os.path.exists(os.path.join(save_path, 'checkpoint')):
        os.makedirs(os.path.join(save_path, 'checkpoint'))
    if not os.path.exists(os.path.join(save_path, 'best_model')):
        os.makedirs(os.path.join(save_path, 'best_model'))

    if not os.path.exists(os.path.join(save_path, 'runs')):
        os.makedirs(os.path.join(save_path, 'runs'))

    if os.path.exists(os.path.join(save_path, 'runs', 'version_0')):
        file_last = sorted(os.listdir(os.path.join(save_path, "runs/")))[-1].split('_')[1]
        writer = SummaryWriter(log_dir= os.path.join(save_path, 'runs/version_') + str(int(file_last) + 1))

    if not os.path.exists(os.path.join(save_path, 'runs', 'version_0')):
        writer = SummaryWriter(log_dir=os.path.join(save_path, 'runs/version_0'))

    for epoch in range(start_epochs, n_epochs + 1):
        train_loss = 0.
        train_error = 0.
        valid_loss = 0.
        valid_error = 0.
        for b in range(num_batches_per_epoch):
            model.train()

            batch = next(tr_gen)
            data = torch.from_numpy(batch['data']).float().to(device)
            label = torch.from_numpy(np.array(batch['label'])).long().to(device)
            bag_label = label[0].long().to(device)
            data, bag_label = Variable(data), Variable(bag_label)

            # reset gradients
            optimizer.zero_grad()
            # calculate loss and metrics
            loss, _ = model.calculate_objective(data, bag_label)
            train_loss += loss.item()
            error, _ = model.calculate_classification_error(data, bag_label)
            train_error += error
            # backward pass
            loss.backward()
            # step
            optimizer.step()
            # meep
            # empty_cache()
        scheduler.step()
        train_loss /= num_batches_per_epoch
        train_error /= num_batches_per_epoch
        print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss, train_error))
        print('here')
        # print(memory_cached())
        writer.add_scalar("Training loss", train_loss, epoch)
        writer.add_scalar("Training error", train_error, epoch)



        ######################
        # validate the model #
        ######################
        for b in range(num_val_batches_per_epoch):

            print('Start Testing')
            model.eval()
            batch = next(val_gen)
            data = torch.from_numpy(batch['data']).float().to(device)
            label = torch.from_numpy(np.array(batch['label'])).long().to(device)
            bag_label = label[0].long().to(device)
            data, bag_label = Variable(data), Variable(bag_label)
            loss, attention_weights = model.calculate_objective(data, bag_label)
            valid_loss += loss.item()
            error, predicted_label = model.calculate_classification_error(data, bag_label)
            valid_error += error


            n = batch['patient']
            label = label[0].cpu().detach().numpy()
            predicted_label = predicted_label[0][0].cpu().detach().numpy()
            # print(n, label, predicted_label)
            # print(str(n[0]) + "\t" + str(label) + "\t" + str(int(predicted_label)))

            print(' Test Set, Loss: {:.4f}, Test error: {:.4f}'.format(valid_loss, valid_error))
        # calculate average losses
        valid_loss /= num_val_batches_per_epoch
        valid_error /= num_val_batches_per_epoch

        writer.add_scalar("Validation loss", valid_loss, epoch)
        writer.add_scalar("Validation error", valid_error, epoch)



        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
        ))

        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        # save checkpoint
        save_ckp(checkpoint, False, os.path.join(save_path, 'checkpoint', checkpoint_path), os.path.join(save_path, 'best_model', best_model_path))

        ## TODO: save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            # save checkpoint as best model
            save_ckp(checkpoint, True, os.path.join(save_path, 'checkpoint', checkpoint_path), os.path.join(save_path, 'best_model', best_model_path))
            valid_loss_min = valid_loss

    # return trained model
    return model



def train_model_cnn(start_epochs, n_epochs, num_batches_per_epoch, num_val_batches_per_epoch, valid_loss_min_input, valid_acc_max_input, model: nn.Module, tr_gen, val_gen, device, optimizer, criterion, checkpoint_path, best_model_path, scheduler, label_name:str):
    """
    Keyword arguments:
    start_epochs -- the real part (default 0.0)
    n_epochs -- the imaginary part (default 0.0)
    valid_loss_min_input
    loaders
    model
    optimizer
    criterion
    use_cuda
    checkpoint_path
    best_model_path

    returns trained model
    """
    # initialize tracker for minimum validation loss and max validation accuracy
    valid_loss_min = valid_loss_min_input
    valid_acc_max = valid_acc_max_input

    save_path = init_model('cnn', path, rel_save=rel_save)
    print('save_path', save_path)
    print(checkpoint_path)
    print('full_path', os.path.join(save_path, 'checkpoint', checkpoint_path))

    #create folder

    # if not os.path.exists(os.path.join(save_path, 'checkpoint')):
    #     os.makedirs(os.path.join(save_path, 'checkpoint'))
    # if not os.path.exists(os.path.join(save_path, 'best_model')):
    #     os.makedirs(os.path.join(save_path, 'best_model'))

    if not os.path.exists(os.path.join(save_path, 'runs')):
        os.makedirs(os.path.join(save_path, 'runs'))

    if not os.path.exists(os.path.join(save_path, 'runs', label_name)):
        os.makedirs(os.path.join(save_path, 'runs', label_name))

    if os.path.exists(os.path.join(save_path, 'runs', label_name, 'version_0')):
        #file_last = sorted(os.listdir(os.path.join(save_path, "runs/")))[-1].split('_')[1]
        file_last = sorted([int(x.split('_')[1]) for x in os.listdir(os.path.join(save_path, "runs", label_name + '/'))])[-1]
        writer = SummaryWriter(log_dir= os.path.join(save_path, 'runs', label_name, 'version_') + str(int(file_last) + 1))
        if not os.path.exists(os.path.join(save_path, 'runs', label_name, 'version_' + str(int(file_last) + 1), 'checkpoint')):
            os.makedirs(os.path.join(save_path, 'runs', label_name, 'version_' + str(int(file_last) + 1), 'checkpoint'))
        if not os.path.exists(os.path.join(save_path, 'runs', label_name, 'version_' + str(int(file_last) + 1), 'best_model')):
            os.makedirs(os.path.join(save_path, 'runs', label_name, 'version_' + str(int(file_last) + 1), 'best_model'))
        final_save_path = os.path.join(save_path, 'runs', label_name, 'version_' + str(int(file_last) + 1))

    if not os.path.exists(os.path.join(save_path, 'runs', label_name, 'version_0')):
        writer = SummaryWriter(log_dir=os.path.join(save_path, 'runs', label_name, 'version_0'))
        if not os.path.exists(os.path.join(save_path, 'runs', label_name, 'version_0', 'checkpoint')):
            os.makedirs(os.path.join(save_path, 'runs', label_name, 'version_0', 'checkpoint'))
        if not os.path.exists(os.path.join(save_path, 'runs', label_name, 'version_0', 'best_model')):
            os.makedirs(os.path.join(save_path, 'runs', label_name, 'version_0', 'best_model'))
        final_save_path = os.path.join(save_path, 'runs', label_name, 'version_0')



    # #
    # for p in model.parameters():  # Set for cradl
    #     p.requires_grad = False
    print(label_name)
    val_dict = dict()
    for epoch in range(start_epochs, n_epochs + 1):
        train_loss = 0.
        valid_loss = 0.
        total = 0
        correct = 0
        prediction = []
        true_value = []
        for b in range(num_batches_per_epoch):
            model.train()

            batch = next(tr_gen)
            data = torch.from_numpy(batch['data']).float().to(device)
            labels = torch.from_numpy(np.array(batch[label_name])).long().to(device)
            # reset gradients
            optimizer.zero_grad()
            # print(data.shape)
            # print(model)

            # #forward pass
            # with torch.no_grad():
            #     data = cradl(images)

            outputs = model(data)
            print(outputs.shape)
            print(labels.shape)
            loss = criterion(outputs, labels)
            loss /= batch_size
            # backward pass
            loss.backward()
            #print(loss)
            train_loss += loss.item()

            # optimizer step
            optimizer.step()
            # meep
            # empty_cache()
        scheduler.step()
        print('Epoch: {}, Loss: {:.4f}'.format(epoch, train_loss))
        print('here')
        # print(memory_cached())
        writer.add_scalar("Training loss", train_loss, epoch)



        ######################
        # validate the model #
        ######################
        for b in range(num_val_batches_per_epoch):

            print('Start Testing')
            model.eval()
            batch = next(val_gen)
            data = torch.from_numpy(batch['data']).float().to(device)
            print('label',label_name)
            labels = torch.from_numpy(np.array(batch[label_name])).long().to(device)
            outputs = model(data)
            print('outputs, label', len(batch['data']), batch[label_name])
            print('outputs', outputs.shape)
            print('labels', labels.shape)

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

            print(' Test Set, Batch_index: {:.0f}, Loss: {:.4f}, Acc: {:.4f}, Correct: {:.0f}, Total: {:.0f}'.format(b, valid_loss/(b+1), 100.*correct/total, correct, total))
        # calculate average losses
        valid_loss /= num_val_batches_per_epoch
        print('total',total)
        valid_acc = 100.*correct/total

        writer.add_scalar("Validation loss", valid_loss, epoch)
        writer.add_scalar("Validation accuracy", valid_acc, epoch)







        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tValidation accuracy: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss,
            valid_acc
        ))

        # create checkpoint variable and add important data
        # acc = 100.*correct/total
        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'valid_acc_max': valid_acc,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        # save checkpoint
        save_ckp(checkpoint, False, os.path.join(final_save_path, 'checkpoint', checkpoint_path), os.path.join(final_save_path, 'best_model', best_model_path))

        ## TODO: save the model if validation accuracy has increased
        #if valid_loss <= valid_loss_min:
        if valid_acc >= valid_acc_max:
            #print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            print('Validation acc increased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_acc_max, valid_acc))

            # save checkpoint as best model
            save_ckp(checkpoint, True, os.path.join(final_save_path, 'checkpoint', checkpoint_path), os.path.join(final_save_path, 'best_model', best_model_path))
            valid_loss_min = valid_loss
            valid_acc_max = valid_acc

            #save metrics
            val_dict['acc']= valid_acc
            # print(true_value)
            # print(prediction)
            # print(np.concatenate(true_value, axis=0))
            # print(np.concatenate(prediction, axis=0))
            if label_name == 'label':
                val_dict['auroc'] = float(roc_auc_score(np.concatenate(true_value, axis=0), np.concatenate(prediction, axis=0)))
                precision, recall, thresholds = precision_recall_curve(np.concatenate(true_value, axis=0), np.concatenate(prediction, axis=0))
                #val_dict['recall'] = float(recall)
                #val_dict['precision'] = float(precision)
                val_dict['auprc'] = float(auc(recall, precision))
                val_dict['valid_loss'] = float(valid_loss)

            #GOLD acc one-off
            elif label_name == 'gold':
                correct_oneoff = 0
                for i in range(0, len(np.concatenate(true_value, axis=0))):
                    if np.concatenate(prediction, axis=0)[i] >= np.concatenate(true_value, axis=0)[i]-1 and np.concatenate(prediction, axis=0)[i] <= np.concatenate(true_value, axis=0)[i]+1:
                        correct_oneoff += 1
                acc_oneoff = 100.*correct_oneoff/total
                val_dict['acc_oneoff']= float(acc_oneoff)
                val_dict['valid_loss'] = float(valid_loss)
            else:
                NotImplemented

            print(val_dict)
            with open(os.path.join(final_save_path, 'best_model', 'val_dict'), 'w+') as file:
                yaml.dump(val_dict, file)
    writer.add_hparams({'batch_size': batch_size, 'transf_prob': prob_sample, 'wd': weight_decay},
                       val_dict, run_name=writer.log_dir)
    # return trained model
    return model

def main_attmech_old(path, resave=False, num_epoch=1):
    dict_output = save_outputs(path, get_slice_idx=False, resave=resave, num_epoch=num_epoch)
    train = reconstruct_vector_attentionmech(dict_output['Train'])
    val = reconstruct_vector_attentionmech(dict_output['Valin'])


    print('Load Train and Test Set')

    dataloader_train = COPDDataloader(train, 1, 1)
    dataloader_val = COPDDataloader(val, 1, 1)

    tr_transforms = get_train_transform()

    tr_gen = MultiThreadedAugmenter(dataloader_train, tr_transforms, num_processes=8,
                                    num_cached_per_queue=3,
                                    seeds=None, pin_memory=False)
    # we need less processes for vlaidation because we dont apply transformations
    val_gen = MultiThreadedAugmenter(dataloader_val, None,
                                     num_processes=max(1, 8 // 2), num_cached_per_queue=1,
                                     seeds=None,
                                     pin_memory=False)

    # lets start the MultiThreadedAugmenter. This is not necessary but allows them to start generating training
    # batches while other things run in the main thread
    tr_gen.restart()
    val_gen.restart()
    batch_1 = next(tr_gen)
    batch_2 = next(tr_gen)
    batch_3 = next(tr_gen)
    batch_4 = next(tr_gen)

    # now if this was a network training you would run epochs like this (remember tr_gen and val_gen generate
    # inifinite examples! Don't do "for batch in tr_gen:"!!!):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Attention().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0005)
    num_batches_per_epoch = 100
    num_val_batches_per_epoch = 100
    num_epochs = 500
    print('Start Training')
    trained_model = train_model_attmech(1, num_epochs, num_batches_per_epoch, num_val_batches_per_epoch, np.Inf, model, tr_gen, val_gen, device, optimizer,
                          "current_checkpoint.pt", "best_model.pt")

def calculate_prob(dict, label):
    print(dict)
    print(label)
    prob_sample = 0
    n_class = list(Counter(dict[label]).values())
    keys_class = list(Counter(dict[label]).keys())
    base_value = 1/len(keys_class)
    print(base_value, n_class)
    for value in keys_class:
        #print(keys_class[value])
        prob_sample += np.where(np.array(dict[label]) == value, base_value / Counter(dict[label])[value], 0)
        # for i in range(0, len(dict['labels'])):
        #     if dict['labels'][i] == keys_class[value]:
        #         prob_sample[i] = base_value/n_class[keys_class[value]]
        #     else:
        #         print(dict['labels'][i])
    print('prob_sample',prob_sample)
    print(sum(prob_sample))
    return prob_sample



def main_attmech(path, resave=False, num_epoch=1, base_train = 'default', learning_rate = 0.001, weight_decay = 3e-5, prob_sample = 0.1):
    #save_outputs(path, get_slice_idx=False, resave=resave, num_epoch=num_epoch)
    dict_train_aux = np.load(os.path.join(path, 'latent_tmp/train_data.npz'))
    dict_valin_aux = np.load(os.path.join(path, 'latent_tmp/valin_data.npz'))
    dict_train = {}
    dict_valin = {}
    for files in dict_train_aux.files:
        dict_train[files] = dict_train_aux[files]
        dict_valin[files] = dict_valin_aux[files]

    dict_output = {'Train': dict_train, 'Valin': dict_valin}
    train = reconstruct_vector_attentionmech(dict_output['Train'])
    val = reconstruct_vector_attentionmech(dict_output['Valin'])


    print('Load Train and Test Set')
    #batch_size = 50

    patch_size = train['latent'][0].shape

    prob_train = calculate_prob(train)
    prob_val = calculate_prob(val)


    print('Load Train and Test Set')

    dataloader_train = COPDDataloader_attmech_unbalanced(train, batch_size = 1, patch_size = patch_size, num_threads_in_multithreaded = 1, sampling_probabilities=prob_train)
    dataloader_val = COPDDataloader_attmech_unbalanced(val, batch_size = 1, patch_size = patch_size, num_threads_in_multithreaded= 1, sampling_probabilities=prob_val)

    tr_transforms = get_train_transform(base_train= base_train, prob_sample= prob_sample)

    tr_gen = MultiThreadedAugmenter(dataloader_train, tr_transforms, num_processes=8,
                                    num_cached_per_queue=3,
                                    seeds=None, pin_memory=False)
    # we need less processes for vlaidation because we dont apply transformations
    val_gen = MultiThreadedAugmenter(dataloader_val, None,
                                     num_processes=max(1, 8 // 2), num_cached_per_queue=1,
                                     seeds=None,
                                     pin_memory=False)

    # lets start the MultiThreadedAugmenter. This is not necessary but allows them to start generating training
    # batches while other things run in the main thread
    tr_gen.restart()
    val_gen.restart()
    batch_1 = next(tr_gen)
    print(batch_1['patient'])
    batch_2 = next(tr_gen)
    batch_3 = next(tr_gen)
    batch_4 = next(tr_gen)

    # now if this was a network training you would run epochs like this (remember tr_gen and val_gen generate
    # inifinite examples! Don't do "for batch in tr_gen:"!!!):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Attention().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    num_batches_per_epoch = 100
    num_val_batches_per_epoch = 100
    num_epochs = 500
    scheduler = CosineAnnealingLR(optimizer, num_epochs, 0.)
    print('Start Training')
    trained_model = train_model_attmech(1, num_epochs, num_batches_per_epoch, num_val_batches_per_epoch, np.Inf, model, tr_gen, val_gen, device, optimizer,
                          "current_checkpoint.pt", "best_model.pt", scheduler)




def main_cnn(path, resave=False, reconstruct= False, num_epoch=1, batch_size = 20, learning_rate = 0.001, weight_decay = 3e-5,
             base_train = 'default', prob_sample = 0.1, model_type = 'linear', classification_type = 'binary',
             input='insp', overlap = '0', realworld_dataset=False):


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

    # train_aux = np.load(os.path.join(path, 'latent_tmp', 'train_reconstructed.npz'), allow_pickle=True)
    # val_aux = np.load(os.path.join(path, 'latent_tmp', 'valin_reconstructed.npz'), allow_pickle=True)
    train_aux = np.load(os.path.join(path, 'latent_tmp/train_cnn_latent', 'train_reconstructed.npz'), allow_pickle=True)
    val_aux = np.load(os.path.join(path, 'latent_tmp/train_cnn_latent', 'valin_reconstructed.npz'), allow_pickle=True)

    train = {}
    val = {}


    for files in train_aux.files:
        train[files] = train_aux[files]
        val[files] = val_aux[files]


    print('Load Train and Test Set')
    #batch_size = 50
    num_channels = train['latent'][0].shape[1]

    patch_size = train['reconstructed'][0][0].shape
    # dataloader_train = COPDDataloader_cnn(train, batch_size, patch_size, 1)
    # dataloader_val = COPDDataloader_cnn(val, batch_size, patch_size, 1)



    prob_train = calculate_prob(train, label_name)
    prob_val = calculate_prob(val, label_name)




    dataloader_train_unbalanced = COPDDataloader_unbalanced(train, batch_size, patch_size, 1, sampling_probabilities=prob_train) #prob_train
    dataloader_val_unbalanced = COPDDataloader_unbalanced(val, batch_size, patch_size, 1, sampling_probabilities=prob_val) #prob_val
    #dataloader_val_unbalanced = COPDDataloader_eval(val, batch_size, patch_size, 1, shuffle=False) #prob_train


    #print('patch_size', patch_size, [i // 2 for i in patch_size])

    transforms_train = get_train_transform_cnn(mode='train',
                                               patch_size=patch_size,
                                               base_train=base_train,
                                               prob_sample = prob_sample)

    tr_gen = MultiThreadedAugmenter(dataloader_train_unbalanced, transforms_train, num_processes=8,
                                    num_cached_per_queue=3,
                                    seeds=None, pin_memory=False)
    # we need less processes for vlaidation because we dont apply transformations
    val_gen = MultiThreadedAugmenter(dataloader_val_unbalanced, None,
                                     num_processes=max(1, 8 // 2), num_cached_per_queue=1,
                                     seeds=None,
                                     pin_memory=False)

    # lets start the MultiThreadedAugmenter. This is not necessary but allows them to start generating training
    # batches while other things run in the main thread
    tr_gen.restart()
    val_gen.restart()
    batch_1 = next(tr_gen)
    # sitk.WriteImage(sitk.GetImageFromArray(batch_1['data'][0][0,:,:,:]), '/home/silvia/Downloads/ex_original_spatial_' + str(batch_1['patient'][0]) + '_0' + '.nii.gz')
    # sitk.WriteImage(sitk.GetImageFromArray(batch_1['data'][0][20,:,:,:]), '/home/silvia/Downloads/ex_original_spatial_'+ str(batch_1['patient'][0])  +'_20' +'.nii.gz')
    # sitk.WriteImage(sitk.GetImageFromArray(batch_1['data'][0][140,:,:,:]), '/home/silvia/Downloads/ex_original_spatial_'+ str(batch_1['patient'][0])  +'_140' +'.nii.gz')
    # sitk.WriteImage(sitk.GetImageFromArray(batch_1['data'][2][0,:,:,:]), '/home/silvia/Downloads/ex_original_spatial_' + str(batch_1['patient'][2]) +'_0' +'.nii.gz')
    # sitk.WriteImage(sitk.GetImageFromArray(batch_1['data'][2][20,:,:,:]), '/home/silvia/Downloads/ex_original_spatial_'+ str(batch_1['patient'][2])  +'_20' +'.nii.gz')
    # sitk.WriteImage(sitk.GetImageFromArray(batch_1['data'][2][140,:,:,:]), '/home/silvia/Downloads/ex_original_spatial_'+ str(batch_1['patient'][2])  +'_140' +'.nii.gz')
    # #from batchviewer import view_batch
    # view_batch(np.transpose(batch_1['data'][0],(3,2,1,0))[0])
    # view_batch(np.transpose(batch_1['data'][0],(3,2,1,0))[20])
    #print(batch_1['data'].shape)

    # view_batch(batch_1['data'][0][0,:,:,:])
    # view_batch(batch_1['data'][0][20,:,:,:])
    # view_batch(batch_1['data'][0][145,:,:,:])
    # view_batch(batch_1['data'][0][231,:,:,:])
    # view_batch(batch_1['data'][0][235,:,:,:])
    # view_batch(batch_1['data'][0][453,:,:,:])

    #learning_rate = 1e-6 #5e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




    if model_type == 'linear':
        model = OnlyLinear(patch_size=patch_size, num_classes= num_classes, channels_in=num_channels).to(device)
    elif model_type == 'resnet18':
        model = ResNet18().to(device)
    elif model_type == 'resnet34':
        model = ResNet34().to(device)
    elif model_type == 'resnet50':
        model = ResNet50().to(device)
    elif model_type == 'LeNet3D':
        model = LeNet3D(num_classes= num_classes, channels_in=num_channels).to(device)
    elif model_type == 'Small_LeNet':
        model = Small_LeNet(num_classes= num_classes, channels_in=num_channels).to(device)
    elif model_type == 'Fully_Connected':
        model = Fully_Connected(num_classes= num_classes, channels_in=num_channels, patch_size=patch_size).to(device)
    else:
        NotImplemented


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    num_epochs = 500
    scheduler = CosineAnnealingLR(optimizer, num_epochs, 0.)

    num_batches_per_epoch = 100
    num_val_batches_per_epoch = 100
    print('Start Training')
    trained_model = train_model_cnn(1, num_epochs, num_batches_per_epoch, num_val_batches_per_epoch, np.Inf, 0, model, tr_gen, val_gen, device, optimizer, criterion,
                          "current_checkpoint.pt", "best_model.pt", scheduler, label_name)

    #print(val_dict)


if __name__ == "__main__":
    args = parser.parse_args()
    path = args.path 
    resave =  args.resave
    reconstruct =  args.reconstruct
    num_epoch = args.num_epoch
    batch_size = args.batch_size
    base_train = args.base_train
    learning_rate = args.lr
    weight_decay = args.wd
    prob_sample = args.probs
    model_type = args.model_type
    exp_type = args.exp_type
    classification_type = args.classification_type
    input = args.input
    overlap = args.overlap
    realworld_dataset = args.realworld_dataset
    print('real_world_analysis', realworld_dataset)

    if exp_type == 'attention':
        main_attmech(path, resave=True, num_epoch=num_epoch, base_train= base_train, learning_rate = learning_rate, weight_decay = weight_decay, prob_sample = prob_sample) #resave
    elif exp_type == 'cnn':
        main_cnn(path, resave=resave, reconstruct = reconstruct, num_epoch=num_epoch, batch_size = batch_size, base_train = base_train, learning_rate = learning_rate,
                 weight_decay = weight_decay, prob_sample = prob_sample, model_type=model_type, classification_type = classification_type,
                 input = input, overlap = overlap, realworld_dataset = realworld_dataset) #resave
    else:
        NotImplemented