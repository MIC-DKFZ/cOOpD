import numpy as np
import pandas as pd
from argparse import ArgumentParser
from config.latent_model import filename, model_dicts, rel_save
import os
import yaml
from metrics.binary_score import get_metrics



parser = ArgumentParser()
parser.add_argument('-p', '--path', type=str, default= '/home/silvia/Documents/CRADL/logs_cradl/copdgene/pretext/brain/nnclr-resnet34/default/20033104')
parser.add_argument('-f', '--folder_eval', type=str, default='eval_split')
parser.add_argument('--name_exp', type=str, default='full')
parser.add_argument('-s', '--split_patients', type=int, default= 1)



def eval_encoder_split(path, version, folder_eval, Model_cls, model_kwargs={'n_components':1}):
    out_dict = dict()
    key = 'nll'
    global_path = os.path.join(path, version, 'INN')#Model_cls.name)
    df_1 = pd.read_csv(os.path.join(global_path, folder_eval + '1', 'lung_val', 'all_info.csv'), index_col=0)
    df_2 = pd.read_csv(os.path.join(global_path, folder_eval + '2', 'lung_val', 'all_info.csv'), index_col=0)
    df_3 = pd.read_csv(os.path.join(global_path, folder_eval + '3', 'lung_val', 'all_info.csv'), index_col=0)

    df = pd.concat([df_1, df_2], axis=0)
    df = pd.concat([df, df_3], axis=0)


    df['label'] = df['label'].map(lambda x: x.lstrip('[').rstrip(']'))
    labels = np.array(df.groupby('patient_name')['label'].apply(lambda grp: np.max(grp))).astype(np.int)

    patient_names = df['patient_name'].to_numpy().flatten()

    val_avg = np.array(df.groupby('patient_name')[key].apply(lambda grp: np.mean(grp)))
    val_median = np.array(df.groupby('patient_name')[key].apply(lambda grp: np.median(grp)))
    val_Q3 = np.array(df.groupby('patient_name')[key].apply(lambda grp: np.quantile(grp, 0.75)))
    val_P95 = np.array(df.groupby('patient_name')[key].apply(lambda grp: np.quantile(grp, 0.95)))
    val_P99 = np.array(df.groupby('patient_name')[key].apply(lambda grp: np.quantile(grp, 0.99)))
    val_Max = np.array(df.groupby('patient_name')[key].apply(lambda grp: np.max(grp)))
    val_SumP95 = np.array(df.groupby('patient_name')[key].apply(lambda grp: np.sum(grp * grp > np.quantile(grp, 0.95))))
    val_SumP99 = np.array(df.groupby('patient_name')[key].apply(lambda grp: np.sum(grp * grp > np.quantile(grp, 0.99))))



    assert labels.shape == val_avg.shape
    out_dict[key + 'mean'] = get_metrics(val_avg, labels)
    out_dict[key + 'median'] = get_metrics(val_median, labels)
    out_dict[key + 'Q3'] = get_metrics(val_Q3, labels)
    out_dict[key + 'P95'] = get_metrics(val_P95, labels)
    out_dict[key + 'P99'] = get_metrics(val_P99, labels)
    out_dict[key + 'Max'] = get_metrics(val_Max, labels)
    out_dict[key + 'SumP95'] = get_metrics(val_SumP95, labels)
    out_dict[key + 'SumP99'] = get_metrics(val_SumP99, labels)

    df.to_csv(os.path.join(global_path, 'lung_val', 'all_info.csv'))
    with open(os.path.join(global_path, 'sample_full.yaml'), 'w+') as file:
        yaml.dump(out_dict, file)





def main(path, version='results_plot_0', name_exp='fast', split_pts = 0, folder_eval='trial_1'):
    for model_dict in model_dicts:
        # print (model_dict
        eval_encoder_split(path=path, version=version, folder_eval=folder_eval, **model_dict)





if __name__ == "__main__":
    args = parser.parse_args()
    path = args.path
    name_exp = args.name_exp
    split_pts = args.split_patients
    folder_eval = args.folder_eval
    main(path, version='results_plot_0', name_exp=name_exp, split_pts=split_pts, folder_eval=folder_eval)