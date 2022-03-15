import os 
from argparse import ArgumentParser

import numpy as np

from pyutils.parser import str2bool
from latent_gen.ood_model import Abstract_OOD
from datamodules.brain_module import BrainDataModule
from algo.model import load_best_model, get_label_latent_forCNN
from config.latent_model import filename, model_dicts, tmp, suffix, rel_save
import torch
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from latent_gen.attention_mech import Attention, COPDDataloader, reconstruct_vector_attentionmech, get_train_transform, MixOrderTransform
import torch.optim as optim
from torch.autograd import Variable
import shutil
from torch.utils.tensorboard import SummaryWriter
import SimpleITK as sitk
from latent_gen.cnn_aftercradl import reconstruct_img, COPDDataloader_cnn, ResNet_Encoder, ResNet18, get_train_transform_cnn
from models import base
import torch.nn as nn
from data_aug.bg_wrapper import get_simclr_pipeline_transform


parser = ArgumentParser()
parser.add_argument('-p', '--path', type=str, default='/home/silvia/Documents/CRADL/logs_cradl/pretext/brain/simclr-VGG13/default/10407366')
parser.add_argument('--num_epoch', type=int, default=1)
parser.add_argument('--resave', type=str2bool, nargs='?', const=True, default=False)


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

def save_outputs(path, resave=False, num_epoch=1, get_slice_idx=True):
    tmp_dir = os.path.join(path, tmp)
    suffix = '_data'
    keys = ['Train', 'Valin']
    print(all([os.path.exists(os.path.join(tmp_dir, key.lower()+suffix+'.npz')) for key in keys]))
    if not (all([os.path.exists(os.path.join(tmp_dir, key.lower()+suffix+'.npz')) for key in keys]) and resave is False):
        experiment, args = load_best_model(path)
        experiment = experiment.to('cuda:0')
        mode='fit'
        if num_epoch != 1:
            mode = 'train'
        datamodule = BrainDataModule(mode=mode, batch_size=64, step = 'attention_mech') #attention_mech #fitting_GMM

        #loader_dict = {'train': datamodule.train_dataloader(), 'val':datamodule.val_dataloader()}
        train_loader, val_loader = datamodule.train_dataloader()
        loader_dict = {'train': train_loader, 'val': val_loader}

        data_dict = dict()
        # from pdb import set_trace as bp 
        # bp()
        for key1, key2 in zip(keys, ['train', 'val']):
            if key1 =='Train':
                data_dict[key1]= get_label_latent_forCNN(experiment, loader_dict[key2], get_slice=get_slice_idx, num_epoch=num_epoch)
            else:
                data_dict[key1]= get_label_latent_forCNN(experiment, loader_dict[key2], get_slice=get_slice_idx)

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
          best_model_path):
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
            print(n, label, predicted_label)
            print(str(n[0]) + "\t" + str(label) + "\t" + str(int(predicted_label)))

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

def train_model_cnn(start_epochs, n_epochs, num_batches_per_epoch, num_val_batches_per_epoch, valid_loss_min_input, model, tr_gen, val_gen, device, optimizer, criterion, checkpoint_path,
          best_model_path):
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

    save_path = init_model('cnn', path, rel_save=rel_save)
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
        valid_loss = 0.
        total = 0
        correct = 0
        for b in range(num_batches_per_epoch):
            model.train()

            batch = next(tr_gen)
            data = torch.from_numpy(batch['data']).float().to(device)
            labels = torch.from_numpy(np.array(batch['label'])).long().to(device)


            #forward pass
            outputs = model(data)
            loss = criterion(outputs, labels)
            # reset gradients
            optimizer.zero_grad()
            # backward pass
            loss.backward()
            print(loss)
            train_loss += loss.item()

            # step
            optimizer.step()
            # meep
            # empty_cache()
        train_loss /= num_batches_per_epoch
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
            labels = torch.from_numpy(np.array(batch['label'])).long().to(device)
            outputs = model(data)

            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()


            n = batch['patient']
            label = labels[0].cpu().detach().numpy()
            print(n, label, correct)
            print(str(n[0]) + "\t" + str(label) + "\t" + str(int(correct)))

            print(' Test Set, Batch_index: {:.0f}, Loss: {:.4f}, Acc: {:.4f}, Correct: {:.0f}, Total: {:.0f}'.format(b, valid_loss/(b+1), 100.*correct/total, correct, total))
        # calculate average losses
        valid_loss /= num_val_batches_per_epoch

        writer.add_scalar("Validation loss", valid_loss, epoch)
        writer.add_scalar("Validation accuracy", 100.*correct/total, epoch)




        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
        ))

        # create checkpoint variable and add important data
        acc = 100.*correct/total
        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'acc': acc,
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

def main_attmech(path, resave=False, num_epoch=1):
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



def main_cnn(path, resave=False, num_epoch=1):
    dict_output = save_outputs(path, get_slice_idx=False, resave=resave, num_epoch=num_epoch)
    train, val = reconstruct_img(dict_output)



    print('Load Train and Test Set')
    batch_size = 20
    num_channels = train['latent'][0].shape[1]
    patch_size = train['reconstructed'][0][0].shape
    dataloader_train = COPDDataloader_cnn(train, batch_size, patch_size, 1)
    dataloader_val = COPDDataloader_cnn(val, batch_size, patch_size, 1)

    transforms = get_train_transform_cnn()

    tr_gen = MultiThreadedAugmenter(dataloader_train, transforms, num_processes=8,
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
    from batchviewer import view_batch
    # view_batch(np.transpose(batch_1['data'][0],(3,2,1,0))[0])
    # view_batch(np.transpose(batch_1['data'][0],(3,2,1,0))[20])
    print(batch_1['data'].shape)

    view_batch(batch_1['data'][0][0,:,:,:])
    view_batch(batch_1['data'][0][20,:,:,:])
    view_batch(batch_1['data'][0][145,:,:,:])
    view_batch(batch_1['data'][0][231,:,:,:])
    view_batch(batch_1['data'][0][235,:,:,:])
    view_batch(batch_1['data'][0][453,:,:,:])

    learning_rate = 1e-6 #5e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = ResNet_Encoder(base_model='resnet18', channels_in=num_channels).to(device)
    model = ResNet18().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=3e-5)

    num_batches_per_epoch = 100
    num_val_batches_per_epoch = 100
    num_epochs = 500
    print('Start Training')
    trained_model = train_model_cnn(1, num_epochs, num_batches_per_epoch, num_val_batches_per_epoch, np.Inf, model, tr_gen, val_gen, device, optimizer, criterion,
                          "current_checkpoint.pt", "best_model.pt")


if __name__ == "__main__":
    args = parser.parse_args()
    path = args.path 
    resave =  args.resave 
    num_epoch = args.num_epoch
    #main_attmech(path, resave=True, num_epoch=num_epoch) #resave
    main_cnn(path, resave=True, num_epoch=num_epoch) #resave
