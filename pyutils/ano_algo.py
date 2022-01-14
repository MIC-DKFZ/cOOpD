import os
import pathlib

from matplotlib import lines

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt 

from algo.utils import get_seg_im_gt, process_batch
from pyutils.plots import visualize_brain_data, visualize_brain_seg_data

import seaborn as sns 
context='paper'
save_format='pdf'
font_scale=2
sns.set_theme(style="whitegrid", context=context, font_scale=2)

 


def visualize_segmentations(batch_dict, score_sample_dict, score_pixel_dict, path, name='Test'):
    """Saves Segmentations to path

    Args:
        batch_dict ([type]): [description]
        score_sample_dict ([type]): [description]
        score_pixel_dict ([type]): [description]
        path ([type]): [description]
        name (str, optional): [description]. Defaults to 'Test'.
    """
    for key in score_pixel_dict.keys():
        for cb in [True, False]:
            use_ind=None
            if 'nll' in score_sample_dict.keys():
                f, a = visualize_brain_data(score_pixel_dict, visualize=key, cb=cb, indices=use_ind, title=None, title_dict=batch_dict)
                f.set_size_inches(12,12)
                print("vis_nll")
            prefix = ''
            if cb:
                prefix= 'cb_'
            plt.savefig(os.path.join(path, '{}{} {}.png'.format(prefix, key, name)))
            plt.close(fig='all')

def helper_score(out_dict, key, val, patient_names):
    out_dict[key] = dict()

    score_names_sample_dict = {}
    print(val)
    print(patient_names)

    for key_n, value_n in zip(patient_names, val):
        if key_n not in score_names_sample_dict:
            score_names_sample_dict[key_n] = [value_n]
        else:
            score_names_sample_dict[key_n].append(value_n)

    print(score_names_sample_dict)
    print(score_names_sample_dict.values())
    print(score_names_sample_dict.keys())
    print(score_names_sample_dict.items())

    dataframe1 = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in score_names_sample_dict.items()]))
    print(dataframe1)

    for k in score_names_sample_dict:
        print(k)
        print(score_names_sample_dict[k])

    avg_score_names_sample_dict = {k: sum(score_names_sample_dict[k]) / len(score_names_sample_dict[k]) for k in
                                   score_names_sample_dict}
    # To get a score by image, uncomment:
    #val = np.array(list(avg_score_names_sample_dict.values()))

    return avg_score_names_sample_dict, val

def visualize_scores(score_sample_dict, batch_dict, path, name='Test'):

    labels = batch_dict['label'].flatten()
    patient_names = batch_dict['patient_name'].flatten()

    names_labels_dict = dict(zip(patient_names, labels))
    #labels without repeated values
    labels = np.array(list(names_labels_dict.values()))


    out_dict = dict()
    for key, scores in score_sample_dict.items():
        sns.set_theme(style="whitegrid", context=context, font_scale=2)
        fig, ax = plt.subplots()
        bins = 25
        label = 'label'
        density = True
        size = (12, 9)
        fig.set_size_inches(size)
        print(batch_dict[label])
        print(batch_dict[label].flatten())
        inliers_complete = scores[batch_dict[label].flatten()==0]
        outliers_complete = scores[batch_dict[label].flatten()==1]

        inliers_wnames, inliers = helper_score(out_dict, key, inliers_complete, patient_names)
        ouliers_wname, outliers = helper_score(out_dict, key, outliers_complete, patient_names)

        x_range = (np.quantile(scores, 0), np.quantile(scores, 0.99))
        thresh = np.quantile(inliers, 0.95)

        ax.hist(inliers, bins, x_range, density=density, color='b', label='Inlier', alpha=0.7)
        ax.hist(outliers, bins, x_range, density=density, color='r', label='Outlier', alpha=0.7)
        ax.set_xlabel('Score')
        if density:
            ax.set_ylabel('Density')
        else:
            ax.set_ylabel('Counts')
        ax.legend(loc='upper left')
        plt.tight_layout()
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join(path, 'hist_{} {}.{}'.format(key, name, save_format)))
        plt.close(fig='all')

        sort_by = 'slice_idxs'
        fig, ax = plt.subplots()
        fig.set_size_inches(size)






    # for key, scores in score_sample_dict.items():
    #     sns.set_theme(style="whitegrid", context=context, font_scale=2)
    #     fig, ax = plt.subplots()
    #     bins = 25
    #     label = 'label'
    #     density = True
    #     size = (12, 9)
    #     fig.set_size_inches(size)
    #     print(batch_dict[label])
    #     print(batch_dict[label].flatten())
    #     inliers = scores[batch_dict[label].flatten()==0]
    #     outliers = scores[batch_dict[label].flatten()==1]
    #     x_range = (np.quantile(scores, 0), np.quantile(scores, 0.99))
    #     thresh = np.quantile(inliers, 0.95)
    #
    #     ax.hist(inliers, bins, x_range, density=density, color='b', label='Inlier', alpha=0.7)
    #     ax.hist(outliers, bins, x_range, density=density, color='r', label='Outlier', alpha=0.7)
    #     ax.set_xlabel('Score')
    #     if density:
    #         ax.set_ylabel('Density')
    #     else:
    #         ax.set_ylabel('Counts')
    #     ax.legend(loc='upper left')
    #     plt.tight_layout()
    #     pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    #     plt.savefig(os.path.join(path, 'hist_{} {}.{}'.format(key, name, save_format)))
    #     plt.close(fig='all')
    #
    #     sort_by = 'slice_idxs'
    #     fig, ax = plt.subplots()
    #     fig.set_size_inches(size)
    #
    #
    #
    #
    #



    # for key, scores in score_sample_dict.items():
    #     sns.set_theme(style="whitegrid", context=context, font_scale=2)
    #     fig, ax = plt.subplots()
    #     bins = 25
    #     label = 'label'
    #     density = True
    #     size = (12, 9)
    #     fig.set_size_inches(size)
    #     print(batch_dict[label])
    #     print(batch_dict[label].flatten())
    #     inliers = scores[batch_dict[label].flatten()==0]
    #     outliers = scores[batch_dict[label].flatten()==1]
    #     x_range = (np.quantile(scores, 0), np.quantile(scores, 0.99))
    #     thresh = np.quantile(inliers, 0.95)
    #
    #     ax.hist(inliers, bins, x_range, density=density, color='b', label='Inlier', alpha=0.7)
    #     ax.hist(outliers, bins, x_range, density=density, color='r', label='Outlier', alpha=0.7)
    #     ax.set_xlabel('Score')
    #     if density:
    #         ax.set_ylabel('Density')
    #     else:
    #         ax.set_ylabel('Counts')
    #     ax.legend(loc='upper left')
    #     plt.tight_layout()
    #     pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    #     plt.savefig(os.path.join(path, 'hist_{} {}.{}'.format(key, name, save_format)))
    #     plt.close(fig='all')
    #
    #     sort_by = 'slice_idxs'
    #     fig, ax = plt.subplots()
    #     fig.set_size_inches(size)



    #     #### Plot of Value vs Slices as LinePlot
    #     fig, ax = plt.subplots()
    #     fig.set_size_inches(size)
    #     df_full = pd.DataFrame(list(zip(batch_dict[sort_by],batch_dict[label], scores)), columns=[sort_by, label, 'Score'] )
    #     df_full[label] = df_full[label].map(lambda x: 'Inlier' if (x==0) else 'Outlier')
    #     sns.lineplot(ax=ax, x=sort_by, y='Score', hue=label, data=df_full, ci='sd')
    #     ax.hlines(y=thresh, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], colors='black', linestyles='dashed', label='Threshold TPR=0.95')
    #     ax.legend(loc='upper right')
    #
    #     ax.set_xlabel('Slice Index')
    #     ax.set_ylabel('Score')
    #     plt.setp(ax.get_xticklabels(),rotation=30,horizontalalignment='right')
    #
    #
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(path, 'sliceScorev2_{} {}.{}'.format(key, name, save_format)))
    #     plt.close(fig='all')
    #
    #
    #     #### Plot of Value vs Slices as LinePlot + Prevalence
    #     sns.set_theme(style="whitegrid", context=context, font_scale=1.5)
    #     fig = plt.figure()
    #     grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.3)
    #     ax_hist = fig.add_subplot(grid[-1, :])
    #     main_ax = fig.add_subplot(grid[:-1, :], sharex=ax_hist)
    #     fig.set_size_inches(size)
    #     df_full = pd.DataFrame(list(zip(batch_dict[sort_by],batch_dict[label], scores)), columns=[sort_by, label, 'Score'] )
    #     df_full[label] = df_full[label].map(lambda x: 'Inlier' if (x==0) else 'Outlier')
    #     sns.lineplot(ax=main_ax, x=sort_by, y='Score', hue=label, data=df_full, ci='sd')
    #     df_full = pd.DataFrame(list(zip(batch_dict[sort_by],batch_dict[label], scores)), columns=[sort_by, label, 'Score'] )
    #     df_full = df_full.groupby(sort_by)[label].mean().reset_index()
    #     width = (np.max(df_full[sort_by]) - np.min(df_full[sort_by]))/(len(df_full[sort_by])*1.2)
    #     ax_hist.bar(x=df_full[sort_by], height =df_full[label], width=width, label='Prevalence', color='gray', alpha=0.7)
    #     ax_hist.set_ylabel('Prevalence')
    #     ax_hist.set_ylim(0, 1)
    #     ax_hist.set_xlabel('Slice Index')
    #     main_ax.hlines(y=thresh, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], colors='black', linestyles='dashed', label='Threshold TPR=0.95')
    #     main_ax.legend(loc='upper left')
    #     main_ax.set_xlabel('')
    #     main_ax.set_ylabel('Score')
    #     plt.savefig(os.path.join(path, 'sliceScorev3_{} {}.{}'.format(key, name, save_format)))
    #     plt.close(fig='all')
    #
    #
    # ## Prevalence Plots
    # sns.set_theme(style="whitegrid", context=context, font_scale=2)
    # fig, ax = plt.subplots()
    # # size = set_size('LLNCS', 2/3)
    # # fig.set_size_inches(size)
    # fig.set_size_inches(size)
    # df = pd.DataFrame(list(zip(batch_dict['slice_idxs'],batch_dict[label])),
    #            columns =['slice_idxs', label])
    # sns.barplot(ax=ax, x='slice_idxs', y=label, data=df, ci=None, color='r', label='Prev. of Slice Index')
    # ax.hlines(y=df[label].mean(), xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], colors='black', linestyles='dashed',
    #                 label='Mean Prev. of Slices')
    # ax.legend(loc='upper right')
    #
    # ax.set_xlabel('Slice Index')
    # ax.set_ylabel('Prevalence')
    # ax.set_ylim((0, 1))
    # # plt.setp(ax.get_xticklabels(),rotation=30,horizontalalignment='right')
    # for i, ax_label in enumerate(ax.get_xticklabels()):
    #     if not (i %8 == 0):
    #         ax_label.set_visible(False)
    # plt.tight_layout()
    # plt.savefig(os.path.join(path, 'prevalence_{}.{}'.format(name, save_format)))
    # plt.close(fig='all')




        



def visualize_gt(path, batch_dict, name):
    """Saves Ground Truth to path

    Args:
        path ([type]): [description]
        batch_dict ([type]): [description]
        name ([type]): [description]
    """
    use_ind = None
    
    f,a = visualize_brain_seg_data(batch_dict, indices=use_ind)
    f.set_size_inches(12,12)
    plt.savefig(os.path.join(path, 'GT {}.png'.format(name)))
    plt.close(fig='all')