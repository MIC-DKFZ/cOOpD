from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torchvision.utils as vutils
from algo.utils import exclude_from_wt_decay, norm, process_batch
from models.contrastive import SimCLR
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from pl_bolts.optimizers.lars import LARS
from pyutils.parser import str2bool


class SimCLR_base(pl.LightningModule):
    def __init__(self, model, hparams, scheduler= True, temperature=0.5, weight_decay=1e-6, learning_rate= 1e-4, z_dim=512, 
        warmup_epochs=5, optim='adam',mlp_norm=False,*args, **kwargs):
        super(SimCLR_base, self).__init__()
        self.z_dim = z_dim
        #self.model = SimCLR(model, z_dim, z_dim//2, batch_norm=mlp_norm)
        self.model = SimCLR(model, z_dim, 512, batch_norm=mlp_norm)

        self.optim=optim
        self.learning_rate = learning_rate
        self.temperature=temperature
        self.weight_decay= weight_decay

        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        # self.hparams = hparams
        self.save_hyperparameters(hparams)
        self.show_input_every = 1000
        self.show_n = 64
        print(self.model)

    def forward(self, x):
        return self.model(x)

    def encode(self, x):
        return self.model.encode(x).flatten(start_dim=1)

    def shared_step(self, batch, sum_samples=True, mode='train', vis=False, ano=False):
        #print(batch)
        #print('reached_batch_form', batch['patient_name'], batch['label'], batch['patch_num'])
        # print('batch_size')
        # print(len(batch['data'][0]))
        # print(len(batch['data'][1]))
        # print(len(batch['label']))

        try:
            from batchviewer import view_batch
            # same patient, two augm
            # first pair
            #print(batch['data'])
            #print(batch['data'][0])

            # view_batch(batch['data'][0][0])
            # view_batch(batch['data'][0][1])



        except ImportError:
            view_batch = None
        #import SimpleITK as sitk
        (x_a, x_b), y = process_batch(batch)
        # print('shapes',x_a.shape)
        # print(x_b.shape, y)


        # print(batch['patient_name'])
        # print(batch['patch_num'])
        # import numpy as np
        # for i in [0, 2, 7]:
        #     numpy_array = np.load('/home/silvia/Documents/CRADL/pre-processed/patches_new_all_overlap0/' + batch['patient_name'][i] + '_' + batch['patch_num'][i] + '.npz',
        #                           mmap_mode="r")
        #     insp = numpy_array['insp']
        #     sitk.WriteImage(sitk.GetImageFromArray(insp), '/home/silvia/Downloads/try_' + str(i) + '.nii.gz')

        try:
            from batchviewer import view_batch
            # same patient, two augm
            # first pair
            # print(x_a.shape)

            # view_batch(batch['data'][0][0])
            # view_batch(batch['data'][0][1])
            # view_batch(x_a[0, 0,:,:,:])
            # view_batch(x_b[0, 0,:,:,:])
            # sitk.WriteImage(sitk.GetImageFromArray(x_a[0, 0,:,:,:].cpu()), '/home/silvia/Downloads/try_0aHU.nii.gz')
            # sitk.WriteImage(sitk.GetImageFromArray(x_b[0, 0,:,:,:].cpu()), '/home/silvia/Downloads/try_0bHU.nii.gz')
            #
            #
            # view_batch(x_a[10, 0, :, :,:])
            # view_batch(x_b[10, 0, :, :,:])
            #
            #
            # view_batch(x_a[2, 0, :, :,:])
            # view_batch(x_b[2, 0, :, :,:])
            # sitk.WriteImage(sitk.GetImageFromArray(x_a[2, 0,:,:,:].cpu()), '/home/silvia/Downloads/try_2aHU.nii.gz')
            # sitk.WriteImage(sitk.GetImageFromArray(x_b[2, 0,:,:,:].cpu()), '/home/silvia/Downloads/try_2bHU.nii.gz')
            #
            # view_batch(x_a[7, 0, :, :,:])
            # view_batch(x_b[7, 0, :, :,:])
            # sitk.WriteImage(sitk.GetImageFromArray(x_a[7, 0,:,:,:].cpu()), '/home/silvia/Downloads/try_7aHU.nii.gz')
            # sitk.WriteImage(sitk.GetImageFromArray(x_b[7, 0,:,:,:].cpu()), '/home/silvia/Downloads/try_7bHU.nii.gz')






        except ImportError:
            view_batch = None

        proj_a, z_a = self.model(x_a)
        proj_b, z_b = self.model(x_b)

        loss = self.nt_xent_loss_fn(proj_a, proj_b)
        if torch.isnan(loss):
            print(batch['patient_name'])
            print('loss nan', loss)
            print(batch['patch_num'])


        loss_dict = {
            f'{mode}/loss':loss
        }

        if vis is True:
            vis_dict = {f'{mode}_i/input': x_a, f'{mode}_j/input': x_b}
            for key, value in vis_dict.items():
                #n=self.show_n
                #print(value.shape)
                #print(value[:64].shape)
                #print(norm(vutils.make_grid(value[:n].to('cpu').detach())))
                #print(list(value.shape))
                #print(list(value.shape)[1])
                # for channel in range(list(value.shape)[1]):
                #     print(channel)
                #     print(value[:,channel,:,:,:].shape)
                #     #vutils.make_grid(value[:n].to('cpu').detach())
                self.logger.experiment.add_image(key, norm(vutils.make_grid(value[:,:,:,:,20].to('cpu').detach())), self.current_epoch,)
        #print('loss',loss, loss_dict)
        return loss, loss_dict


    def training_step(self, batch, batch_idx, *args, **kwargs):
        mode = 'train'
        vis = bool(batch_idx%self.show_input_every==0)

        loss, loss_dict = self.shared_step(batch, sum_samples=True, mode=mode, vis=vis)
        self.log_dict(loss_dict, on_step=False, on_epoch=True)
        if torch.isnan(loss):
            print(batch['patient_name'])
            print('loss nan',loss)
            print(batch['patch_num'])
        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        mode = 'val'
        vis = bool(batch_idx == 0)
        loss, out_dict = self.shared_step(batch, mode=mode, vis=vis)
        if torch.isnan(loss):
            print(batch['patient_name'])
            print('loss nan',loss)
            print(batch['patch_num'])
        self.log_dict(out_dict, on_step=False, on_epoch=True)


    def test_step(self, batch, batch_idx, *args, **kwargs):
        mode = 'test'
        vis = bool(batch_idx == 0)
        loss, out_dict = self.shared_step(batch, mode=mode, vis=vis)
        self.log_dict(out_dict, on_step=False, on_epoch=True)

    def nt_xent_loss_fn(self, z1, z2):
        return self.nt_xent_loss(z1, z2, temperature=self.temperature, norm=True)

    @staticmethod
    def nt_xent_loss(z1, z2, temperature, norm=True, sum_samples=True):
        out = torch.cat([z1, z2], dim=0)
        n_z1 = z1.size(0)
        n_samples = len(out)
        if norm is True:
            normalizer = torch.sum(out**2, dim=-1, keepdim=True)**0.5      
            normalizer[normalizer==0] = 1
            out = out/normalizer

        cov = torch.mm(out, out.t().contiguous())
        sim_exp = cov/temperature 

        mask = ~torch.eye(n_samples, device=sim_exp.device).bool()
        neg_log = sim_exp.masked_select(mask).view(n_samples, -1).logsumexp(dim=-1)

        pos_log = torch.sum(out[:n_z1] * out[n_z1:], dim=-1)/temperature
        pos_log = torch.cat([pos_log, pos_log], dim=0)
        loss = -(pos_log - neg_log)
        if sum_samples is True:
            loss = loss.mean()
        return loss

    def setup(self, stage):
        global_batch_size = self.trainer.world_size * self.hparams.batch_size
        self.train_iters_per_epoch = self.hparams.num_samples // global_batch_size

    def configure_optimizers(self):
        # b1 = 0.95
        # b2 = 0.99
        params = exclude_from_wt_decay(self.named_parameters(), weight_decay=self.weight_decay)
        # opt = torch.optim.Adam(params, lr=self.learning_rate, betas=(b1, b2))
        if self.optim=='adam':
            opt = torch.optim.Adam(params, lr=self.learning_rate, betas=(0.9, 0.999))
        elif self.optim=='lars':
            opt = LARS(
                params,
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
                trust_coefficient=0.001,
            )
        else:
            raise NotImplementedError
        if self.scheduler:
            warm_steps = self.warmup_epochs * self.train_iters_per_epoch
            max_steps = self.trainer.max_epochs * self.train_iters_per_epoch
            #max_steps = self.hparams.max_steps
            lin_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer=opt,
                    lr_lambda=linear_warmup_decay(warm_steps, max_steps, cosine=True),
                )

            scheduler = {
                'scheduler' : lin_scheduler,
                'interval' : 'step',
                'frequency' : 1,
            }
            return [opt], [scheduler]
        return [opt], []

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", default=1e-4, type=float)
        parser.add_argument("--num_ft", default=16, type=int)
        parser.add_argument("--z_dim", default=512, type=int) #20
        parser.add_argument("--scheduler", default=True, type=str2bool, const=True, nargs='?')
        parser.add_argument("--temperature", default=0.5, type=float)
        parser.add_argument("--weight_decay", default=1e-6, type=float)
        parser.add_argument("--warmup_epochs", default=5, type=int)
        parser.add_argument("--model_type", choices=['VGG11', 'VGG13', 'VGG16', 'VGG19', 'resnet18', 'resnet34', 'resnet50','CNN3D'], default= 'resnet18', type=str) #CNN3D #resnet18 #VGG11 #VGG13 #VGG16 #VGG19
        parser.add_argument("--mlp_norm", default=False, type=str2bool, const=True, nargs='?')
        parser.add_argument("--augmentation", choices=['standard', 'standard-rot','standard-blur', 'ce', 'ce-blur', 'ce-no_crop', 'random_crop', 'random_crop-ce'],default='random_crop', type=str)

        return parser
