from copy import deepcopy

import numpy as np

from batchgenerators.transforms import BrightnessMultiplicativeTransform, BrightnessTransform, GaussianNoiseTransform, \
    MirrorTransform, SpatialTransform
from batchgenerators.transforms.abstract_transforms import Compose, RndTransform
from batchgenerators.transforms.color_transforms import ClipValueRange
from batchgenerators.transforms.crop_and_pad_transforms import CenterCropTransform, PadTransform #, FillupPadTransform
from batchgenerators.transforms.noise_transforms import BlankSquareNoiseTransform, GaussianBlurTransform 
from batchgenerators.transforms.spatial_transforms import ResizeTransform, ZoomTransform
from batchgenerators.transforms.utility_transforms import AddToDictTransform, CopyTransform, NumpyToTensor, \
    ReshapeTransform
from batchgenerators.transforms.abstract_transforms import AbstractTransform



class DoubleHeadedTransform(AbstractTransform):
    def __init__(self, transform_i, transform_j):
        super().__init__()
        self.transform_i = transform_i 
        self.transform_j = transform_j

    def __call__(self, **data_dict):
        out_dict_1 = self.transform_i(**data_dict)
        out_dict_2 = self.transform_j(**data_dict)
        out_dict_1['data'] = (out_dict_1['data'], out_dict_2['data'])
        # out_dict_1['data_2'] = out_dict_2['data']
        return out_dict_1
    
    def __repr__(self):
        return str(type(self).__name__) + " \t ( " + repr(self.transform_i) + "\n\t" + repr(self.transform_i)+ " )"

class SplitHeadTransformation(AbstractTransform):
    def __init__(self, transform_base, transform_split):
        super().__init__()
        self.transform_base = transform_base
        self.transform_split = transform_split
        self.final_transform = NumpyToTensor()

    def __call__(self, **data_dict):
        out_dict_base = self.transform_base(**data_dict)
        out_dict_split = self.transform_split(**{'data':deepcopy(out_dict_base['data'])})
        # out_dict_split = self.transform_base(**{'data':data_dict['data']})
        out_dict_base['data'] = (self.final_transform(**out_dict_base)['data'], self.final_transform(**out_dict_split)['data'])
        return out_dict_base

    def __repr__(self):
        return str(type(self).__name__) + " \t ( " + repr(self.transform_base) + "\n\t" + repr(self.transform_base)+repr(self.transform_split)+ " )"

class RandomIntensityScale(AbstractTransform):

    def __init__(self, prob:float=0.5, factors: tuple=(-1, -1)):
        """Multiplies the intensity of an input with random values drawn in the range factors.
        

        Args:
            prob ([type]): [Probability for the transformation]
            factors (tuple, optional): [Range of values from which factors are drawn]. Defaults to (-1, -1).
        """
        super().__init__()
        self.prob= prob 
        self.factors = factors
        

    def __call__(self, **data_dict):
        if np.random.uniform()<= self.prob:
            data_dict['data'] = data_dict['data']*np.random.uniform(self.factors[0], self.factors[1])
        return data_dict

def get_transforms(mode="train", target_size=128, add_noise=False, mask_type="", base_train='default',
                   rotate=True, elastic_deform=True, rnd_crop=False, color_augment=True,
                   add_transforms=(), double_headed=False, transform_type='single'):
    transform_list = []
    noise_list = []

    if isinstance(target_size, int):
        target_size1 = target_size2 = target_size
    elif isinstance(target_size, (list, tuple)):
        target_size1 = target_size[0]
        target_size2 = target_size[1]
    else:
        raise NotImplementedError

    if mode == "train":
        if base_train == 'default':
            transform_list = [#FillupPadTransform(min_size=(target_size1 + 5, target_size2 + 5)),
                            ResizeTransform(target_size=(target_size1 + 1, target_size2 + 1),
                                            order=1, concatenate_list=True),

                            # RandomCropTransform(crop_size=(target_size + 5, target_size + 5)),
                            MirrorTransform(axes=(2,)),
                            # ReshapeTransform(new_shape=(1, -1, "h", "w")),
                            SpatialTransform(patch_size=(target_size1, target_size2), random_crop=rnd_crop,
                                            patch_center_dist_from_border=(target_size1 // 2, target_size2 // 2),
                                            do_elastic_deform=elastic_deform, alpha=(0., 100.), sigma=(10., 13.),
                                            do_rotation=rotate,
                                            angle_x=(-0.1, 0.1), angle_y=(0, 1e-8), angle_z=(0, 1e-8),
                                            scale=(0.9, 1.2),
                                            border_mode_data="nearest", border_mode_seg="nearest"),
                            # ReshapeTransform(new_shape=(batch_size, -1, "h", "w"))
                            ]
        elif base_train == 'random_crop':
            rnd_crop = True
            transform_list = [#FillupPadTransform(min_size=(target_size1 + 5, target_size2 + 5)),
                            ResizeTransform(target_size=(target_size1 + 1, target_size2 + 1),
                                            order=1, concatenate_list=True),

                            # RandomCropTransform(crop_size=(target_size + 5, target_size + 5)),
                            MirrorTransform(axes=(2,)),
                            # ReshapeTransform(new_shape=(1, -1, "h", "w")),
                            SpatialTransform(patch_size=(target_size1, target_size2), random_crop=rnd_crop,
                                            patch_center_dist_from_border=(target_size1 // 3, target_size2 // 3),
                                            do_elastic_deform=elastic_deform, alpha=(0., 100.), sigma=(10., 13.),
                                            do_rotation=rotate,
                                            angle_x=(-0.1, 0.1), angle_y=(0, 1e-8), angle_z=(0, 1e-8),
                                            scale=(0.9, 1.2),
                                            border_mode_data="nearest", border_mode_seg="nearest"),
                            # ReshapeTransform(new_shape=(batch_size, -1, "h", "w"))
                            ]
        else:
            raise NotImplementedError
        if color_augment:
            transform_list += [  # BrightnessTransform(mu=0, sigma=0.2),
                BrightnessMultiplicativeTransform(multiplier_range=(0.95, 1.1))]

        transform_list += [
            GaussianNoiseTransform(noise_variance=(0., 0.05)),
            ClipValueRange(min=-1.5, max=1.5),
            # CopyTransform({"data": ("data_clean", "data_blur")}, copy=True),
            # GaussianBlurTransform(blur_sigma=3.5, data_key="data_blur")
        ]

        noise_list = []  # [GaussianNoiseTransform(noise_variance=(0., 0.2))]

        if mask_type == "blank":
            noise_list += [BlankSquareNoiseTransform(squre_size=(target_size // 2),
                                                     noise_val=0,
                                                     n_squres=1,
                                                     square_pos=[(0, 0), (target_size // 2, 0), (0, target_size // 2),
                                                                 (target_size // 2, target_size // 2)])]
        elif mask_type == "context":
            noise_list += [BlankSquareNoiseTransform(squre_size=(0, target_size // 2),
                                                     noise_val=(-1.5, +1.5),
                                                     n_squres=1,)]
        elif mask_type == 'context_david':
            noise_list += [BlankSquareNoiseTransform(squre_size=(0, target_size // 2),
                                                     noise_val=(-1.5, +1.5),
                                                     n_squres=1,)]
        elif mask_type == "noise":
            noise_list += [BlankSquareNoiseTransform(squre_size=(0, target_size // 2),
                                                     noise_val=(-1.5, +1.5),
                                                     n_squres=(0, 3),
                                                     channel_wise_n_val=True)]
        elif mask_type == "test":
            pass
            # noise_list += [SquareMaskTransform(squre_size=(0, np.max(target_size) // 2),
            #                                    noise_val=(-1.5, +1.5),
            #                                    n_squres=(0, 3),
            #                                    channel_wise_n_val=True)]

        elif mask_type == 'noise_rnd_scale':
            noise_list += [RandomIntensityScale(), BlankSquareNoiseTransform(squre_size=(0, target_size // 2),
                                                     noise_val=(-1.5, +1.5),
                                                     n_squres=(0, 3),
                                                     channel_wise_n_val=True)]

        elif mask_type == "gaussian":
            noise_list += [GaussianNoiseTransform(noise_variance=(0., 0.2))]

        elif isinstance(mask_type, float):
            noise_list += [RndTransform(transform=Compose([BlankSquareNoiseTransform(squre_size=(0, target_size // 2),
                                                                                     noise_val=(-1.5, +1.5),
                                                                                     n_squres=(0, 3),
                                                                                     channel_wise_n_val=False),
                                                           AddToDictTransform(in_key="has_no_noise", in_val=0)]),
                                        prob=mask_type,
                                        alternative_transform=AddToDictTransform(in_key="has_no_noise", in_val=1))]

    elif mode == "val":

        transform_list = [#FillupPadTransform(min_size=(target_size1 + 5, target_size2 + 5)),
                         ResizeTransform(target_size=(target_size1 + 1, target_size2 + 1),
                                         order=1, concatenate_list=True),
                         CenterCropTransform(crop_size=(target_size1, target_size2)),
                         ClipValueRange(min=-1.5, max=1.5),
                         # BrightnessTransform(mu=0, sigma=0.2),
                         # BrightnessMultiplicativeTransform(multiplier_range=(0.95, 1.1)),
                         # CopyTransform({"data": "data_clean"}, copy=True)
                         ]

        # noise_list = [CopyTransform({"data": ("data_clean", "data_blur")}, copy=True)]
        #
        # if mask_type == "blank":
        #     noise_list += [CopyTransform({"data": ("data_mask_1", "data_mask_2",
        #                                            "data_mask_3", "data_mask_4")}, copy=True),
        #                    BlankSquareNoiseTransform(squre_size=(target_size // 2), noise_val=0, n_squres=1,
        #                                              square_pos=[(0, 0)], data_key="data_mask_1"),
        #                    BlankSquareNoiseTransform(squre_size=(target_size // 2), noise_val=0, n_squres=1,
        #                                              square_pos=[(target_size // 2, 0)], data_key="data_mask_2"),
        #                    BlankSquareNoiseTransform(squre_size=(target_size // 2), noise_val=0, n_squres=1,
        #                                              square_pos=[(0, target_size // 2)], data_key="data_mask_3"),
        #                    BlankSquareNoiseTransform(squre_size=(target_size // 2), noise_val=0, n_squres=1,
        #                                              square_pos=[(target_size // 2, target_size // 2)],
        #                                              data_key="data_mask_4")]
        #
        # noise_list += [  # GaussianNoiseTransform(noise_variance=(0.1, 0.2)),
        #     GaussianBlurTransform(blur_sigma=1.5, data_key="data_blur")]

    if transform_type == 'split':
        final_transform = SplitHeadTransformation(Compose(transform_list), Compose(noise_list))
        return final_transform
    if add_noise:
        transform_list = transform_list + noise_list

    # if add_resize:
    #     resize_list = [
    #         CopyTransform({"seg": "label"}, copy=True),
    #
    #         CopyTransform({"data": "data_1"}, copy=True),
    #         ZoomTransform(zoom_factors=(1, 1, 0.5, 0.5), order=1),
    #         CopyTransform({"data": "data_2"}, copy=True),
    #         ZoomTransform(zoom_factors=(1, 1, 0.5, 0.5), order=1),
    #         CopyTransform({"data": "data_3"}, copy=True),
    #         ZoomTransform(zoom_factors=(1, 1, 0.5, 0.5), order=1),
    #         CopyTransform({"data": "data_4"}, copy=True),
    #         CopyTransform({"data_1": "data"}, copy=True),
    #
    #         CopyTransform({"data_clean": "data_clean_1"}, copy=True),
    #         ZoomTransform(zoom_factors=(1, 1, 0.5, 0.5), order=1, data_key="data_clean"),
    #         CopyTransform({"data_clean": "data_clean_2"}, copy=True),
    #         ZoomTransform(zoom_factors=(1, 1, 0.5, 0.5), order=1, data_key="data_clean"),
    #         CopyTransform({"data_clean": "data_clean_3"}, copy=True),
    #         ZoomTransform(zoom_factors=(1, 1, 0.5, 0.5), order=1, data_key="data_clean"),
    #         CopyTransform({"data_clean": "data_clean_4"}, copy=True),
    #         CopyTransform({"data_clean_1": "data_clean"}, copy=True),
    #     ]
    #
    #     tranform_list = tranform_list + resize_list

    if transform_type == 'split':
        final_transform = SplitHeadTransformation(Compose(transform_list), Compose(noise_list))
        return final_transform
    if add_noise:
        transform_list = transform_list + noise_list

    transform_list += list(add_transforms)

    transform_list.append(NumpyToTensor())

    final_transform = Compose(transform_list)
    if double_headed:
        final_transform = DoubleHeadedTransform(final_transform, final_transform)
    return final_transform