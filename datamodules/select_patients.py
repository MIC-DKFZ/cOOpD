import os

import numpy as np
import random
from batchgenerators.utilities.file_and_folder_operations import *
import pandas as pd
import json
from sklearn.model_selection import KFold, train_test_split

def select_patients_fortasks(data_folder):
    #old
    #npy_files = subfiles(data_folder[0], suffix=".npz", join=True)
    #new
    list_filenames=[]
    all_patches = [_ for _ in os.listdir(data_folder[0].replace('no_resample/','patches_all_overlap0')) if _.endswith('.npz')]
    all_patches_unique = [i.split('_', 1)[0] for i in all_patches]
    npy_files = list(set(all_patches_unique))

    # remove npy file extension
    patients = [str(os.path.basename(i)).split('.')[0] for i in npy_files]

    annotation = pd.read_csv(
        os.path.join(data_folder[0], 'COPD_criteria_complete.csv'),
        sep=',', converters={'patient': lambda x: str(x)})

    annotation = annotation[annotation.notna()]
    annotation = annotation.dropna(subset=["condition_COPD_GOLD"])
    list_patients_low = [x.lower() for x in patients]
    dir_csv = annotation['patient'].to_list()
    dir_csv = [x.lower() for x in dir_csv]
    print('attention these npz files dont have labels:', list(set(list_patients_low).difference(dir_csv)))
    print('move npz file')

    healthy = annotation.loc[annotation['condition_COPD_GOLD'] == 0]
    COPD = annotation.loc[annotation['condition_COPD_GOLD'] == 1]

    dir_csv_healthy = healthy['patient'].to_list()
    dir_csv_healthy = [x.lower() for x in dir_csv_healthy]

    print(list(set(list_patients_low).intersection(dir_csv_healthy)))
    print(len(list(set(list_patients_low).intersection(dir_csv_healthy))))

    healthy_list = list(set(list_patients_low).intersection(dir_csv_healthy))
    print(healthy_list)
    sizes_healthy = [int(len(healthy_list)*0.5), int(len(healthy_list)*0.25), int(len(healthy_list)*0.25)]
    random.shuffle(healthy_list)
    print(healthy_list)
    it = iter(healthy_list)
    healthy_list_separated = [[next(it) for _ in range(size)] for size in sizes_healthy]
    healthy_pretext_GMM_list = healthy_list_separated[0]
    healthy_eval_list = healthy_list_separated[1]
    healthy_testset_list = healthy_list_separated[2]

    dir_csv_copd = COPD['patient'].to_list()
    dir_csv_copd = [x.lower() for x in dir_csv_copd]

    print(list(set(list_patients_low).intersection(dir_csv_copd)))
    print(len(list(set(list_patients_low).intersection(dir_csv_copd))))

    copd_list = list(set(list_patients_low).intersection(dir_csv_copd))

    sizes_copd = [int(len(copd_list)*0.5), int(len(copd_list)*0.25), int(len(copd_list)*0.25)]
    random.shuffle(copd_list)
    it = iter(copd_list)
    copd_list_separated = [[next(it) for _ in range(size)] for size in sizes_copd]
    copd_pretext_GMM_list = copd_list_separated[0]
    copd_eval_list = copd_list_separated[1]
    copd_testset_list = copd_list_separated[2]


    #copd_pretext_list = random.choices(copd_list, k=200)


    print('final for training')
    print('num_copd_random', len(copd_pretext_GMM_list))
    print('num_healthy_pretextGMM', len(healthy_pretext_GMM_list))
    print('num_healthy_eval', len(healthy_eval_list))
    print('num_healthy_testset', len(healthy_testset_list))
    print('num_copd_eval', len(copd_eval_list))
    print('copd_testset', len(copd_testset_list))


    filenames_for_pretext = copd_pretext_GMM_list + healthy_pretext_GMM_list
    filenames_for_GMM = healthy_pretext_GMM_list
    filenames_for_eval = copd_eval_list + healthy_eval_list
    filenames_for_testset = copd_testset_list + healthy_testset_list
    print(os.path.join(data_folder[0].replace('no_resample','all_no_resample/')))
    #saves the list of patients in txt file
    with open(os.path.join(data_folder[0].replace('no_resample','all_no_resample/'), 'filenames_for_pretext.txt'), 'w') as fp:
        json.dump(filenames_for_pretext, fp)
    with open(os.path.join(data_folder[0].replace('no_resample','all_no_resample/'), 'filenames_for_GMM.txt'), 'w') as fp:
        json.dump(filenames_for_GMM, fp)
    with open(os.path.join(data_folder[0].replace('no_resample','all_no_resample/'), 'filenames_for_eval.txt'), 'w') as fp:
        json.dump(filenames_for_eval, fp)
    with open(os.path.join(data_folder[0].replace('no_resample','all_no_resample/'), 'filenames_for_testset.txt'), 'w') as fp:
        json.dump(filenames_for_testset, fp)
    with open(os.path.join(data_folder[0].replace('no_resample', 'all_no_resample/'),
                           'filenames_copd.txt'), 'w') as fp:
        json.dump(copd_pretext_GMM_list + copd_eval_list + copd_testset_list, fp)
    with open(os.path.join(data_folder[0].replace('no_resample', 'all_no_resample/'),
                           'filenames_healthy.txt'), 'w') as fp:
        json.dump(healthy_pretext_GMM_list + healthy_eval_list + healthy_testset_list, fp)




def select_patients_fortasks_wrongemphysema(data_folder):
    annotation = pd.read_csv(
        os.path.join(data_folder[0].replace('/pre-processed/no_resample', ''), 'all_info_patches_COPD.csv'),
        sep=',', converters={'patient': lambda x: str(x)})

    annotation = annotation[annotation.notna()]
    annotation.dropna(axis=0, how='any', inplace=True)

    healthy = annotation.loc[annotation['emph_0.01'] == 0]
    COPD = annotation.loc[annotation['emph_0.01'] == 1]

    healthy_list = healthy['patch_name'].to_list()
    healthy_list = [x.lower() for x in healthy_list]
    print(healthy_list)

    sizes_healthy = [int(len(healthy_list)*0.5), int(len(healthy_list)*0.25), int(len(healthy_list)*0.25)]
    random.shuffle(healthy_list)
    print(healthy_list)
    it = iter(healthy_list)
    healthy_list_separated = [[next(it) for _ in range(size)] for size in sizes_healthy]
    healthy_pretext_GMM_list = healthy_list_separated[0]
    healthy_eval_list = healthy_list_separated[1]
    healthy_testset_list = healthy_list_separated[2]

    copd_list = COPD['patch_name'].to_list()
    copd_list = [x.lower() for x in copd_list]

    sizes_copd = [int(len(copd_list)*0.5), int(len(copd_list)*0.25), int(len(copd_list)*0.25)]
    random.shuffle(copd_list)
    it = iter(copd_list)
    copd_list_separated = [[next(it) for _ in range(size)] for size in sizes_copd]
    copd_pretext_GMM_list = copd_list_separated[0]
    copd_eval_list = copd_list_separated[1]
    copd_testset_list = copd_list_separated[2]


    #copd_pretext_list = random.choices(copd_list, k=200)


    print('final for training')
    print('num_copd_pretext', len(copd_pretext_GMM_list))
    print('num_healthy_pretextGMM', len(healthy_pretext_GMM_list))
    print('num_healthy_eval', len(healthy_eval_list))
    print('num_healthy_testset', len(healthy_testset_list))
    print('num_copd_eval', len(copd_eval_list))
    print('copd_testset', len(copd_testset_list))


    filenames_for_pretext = copd_pretext_GMM_list + healthy_pretext_GMM_list
    filenames_for_GMM = healthy_pretext_GMM_list
    filenames_for_eval = copd_eval_list + healthy_eval_list
    filenames_for_testset = copd_testset_list + healthy_testset_list
    print(os.path.join(data_folder[0].replace('no_resample','all_no_resample/')))
    #saves the list of patients in txt file
    with open(os.path.join(data_folder[0].replace('no_resample','all_no_resample/'), 'patches_for_pretext.txt'), 'w') as fp:
        json.dump(filenames_for_pretext, fp)
    with open(os.path.join(data_folder[0].replace('no_resample','all_no_resample/'), 'patches_for_GMM.txt'), 'w') as fp:
        json.dump(filenames_for_GMM, fp)
    with open(os.path.join(data_folder[0].replace('no_resample','all_no_resample/'), 'patches_for_eval.txt'), 'w') as fp:
        json.dump(filenames_for_eval, fp)
    with open(os.path.join(data_folder[0].replace('no_resample','all_no_resample/'), 'patches_for_testset.txt'), 'w') as fp:
        json.dump(filenames_for_testset, fp)
    with open(os.path.join(data_folder[0].replace('no_resample', 'all_no_resample/'),
                           'patches_copd.txt'), 'w') as fp:
        json.dump(copd_pretext_GMM_list + copd_eval_list + copd_testset_list, fp)
    with open(os.path.join(data_folder[0].replace('no_resample', 'all_no_resample/'),
                           'patches_healthy.txt'), 'w') as fp:
        json.dump(healthy_pretext_GMM_list + healthy_eval_list + healthy_testset_list, fp)

def select_patients_fortasks_emphysema_withfolds(data_folder, overlap, dataset, n_folds):
    if dataset == 'cosyconet':
        patient = 'patient'
        fev = 'FEV1_GLI'
        fvc = 'FVC_GLI'
        gold = 'GOLD_gli'
        gold_threshold = 1
    elif dataset == 'copdgene':
        patient = 'SUBJECT_ID'
        fev = 'FEV1_post'
        fvc = 'FVC_post'
        gold = 'finalGold'
        gold_threshold = 0

    annotation = pd.read_csv(
        os.path.join(data_folder[0].replace('/pre-processed', ''), 'all_info_patches_COPD' + overlap + '.csv'),
        sep=',', converters={patient: lambda x: str(x)})

    #annotation = annotation[annotation.notna()]
    #annotation.dropna(axis=0, how='any', inplace=True)


    ######
    annotation_drop = annotation.drop_duplicates(subset=[patient])
    healthy = annotation.loc[annotation[gold] <= gold_threshold]
    copd = annotation.loc[annotation[gold] > gold_threshold]

    healthy_noemph = healthy[healthy['emph_0.01'] == 0]
    X = list(annotation_drop[patient])
    y = np.where(annotation_drop[gold] >= 1, 1, 0).tolist()
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    y_test = np.array(y_test)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=100)
    fold = 1
    for train_index, test_index in kf.split(X_train_val):
        if not os.path.exists(os.path.join(data_folder[0], 'overlap' + overlap, 'fold' + str(fold))):
            os.makedirs(os.path.join(data_folder[0], 'overlap' + overlap, 'fold' + str(fold)))
        X_train, X_val, y_train, y_val = np.array(X_train_val)[train_index], np.array(X_train_val)[test_index], np.array(y_train_val)[train_index], np.array(y_train_val)[test_index]
        print(X_train, X_val, y_train, y_val)
        print('final for training')
        print('num_pretext', len(X_train))
        print('num_pretext_copd', len(y_train[y_train>0]))
        print('num_pretext_healthy', len(y_train[y_train<=0]))
        print('num_eval', len(X_val))
        print('num_val_copd', len(y_val[y_val>0]))
        print('num_val_healthy', len(y_val[y_val<=0]))
        print('num_test', len(X_test))
        print('num_test_copd', len(y_test[y_test>0]))
        print('num_test_healthy', len(y_test[y_test<=0]))

        patches_pretext_list = [p for i in X_train for p in annotation['patch_name'].to_list() if p.startswith(i)]
        patches_GMM_list = [p for i in X_train for p in healthy_noemph['patch_name'].to_list() if p.startswith(i)]
        patches_validation_list = [p for i in X_val for p in annotation['patch_name'].to_list() if p.startswith(i)]

        with open(os.path.join(data_folder[0], 'overlap' + overlap, 'fold' + str(fold), 'patches_for_pretext.txt'), 'w') as fp:
            json.dump(patches_pretext_list, fp)
        with open(os.path.join(data_folder[0], 'overlap' + overlap, 'fold' + str(fold), 'patches_for_GMM.txt'), 'w') as fp:
            json.dump(patches_GMM_list, fp)
        with open(os.path.join(data_folder[0], 'overlap' + overlap, 'fold' + str(fold), 'patches_for_eval.txt'), 'w') as fp:
            json.dump(patches_validation_list, fp)


        fold+=1

    patches_test_list = [p for i in X_test for p in annotation['patch_name'].to_list() if p.startswith(i)]
    with open(os.path.join(data_folder[0], 'overlap' + overlap, 'patches_for_testset.txt'),'w') as fp:
        json.dump(patches_test_list, fp)

    with open(os.path.join(data_folder[0], 'overlap' + overlap, 'patches_copd.txt'), 'w') as fp:
        json.dump(copd['patch_name'].to_list() , fp)
    with open(os.path.join(data_folder[0], 'overlap' + overlap, 'patches_healthy.txt'), 'w') as fp:
        json.dump(healthy['patch_name'].to_list() , fp)
    ######

def select_patients_fortasks_emphysema(data_folder, overlap, dataset):
    if dataset == 'cosyconet':
        patient = 'patient'
        fev = 'FEV1_GLI'
        fvc = 'FVC_GLI'
        gold = 'GOLD_gli'
        gold_threshold = 1
    elif dataset == 'copdgene':
        patient = 'SUBJECT_ID'
        fev = 'FEV1_post'
        fvc = 'FVC_post'
        gold = 'finalGold'
        gold_threshold = 0

    annotation = pd.read_csv(
        os.path.join(data_folder[0].replace('/pre-processed', ''), 'all_info_patches_COPD' + overlap + '.csv'),
        sep=',', converters={patient: lambda x: str(x)})

    # annotation = annotation[annotation.notna()]
    # annotation.dropna(axis=0, how='any', inplace=True)
    healthy = annotation.loc[annotation[gold] < gold_threshold]
    COPD = annotation.loc[annotation[gold] >= gold_threshold]

    #healthy_list = healthy['patient'].to_list()
    healthy_list = list(set(healthy[patient]))

    healthy_list = [x for x in healthy_list] #[x.lower() for x in healthy_list]
    #remove duplicates
    healthy_list = list(set(healthy_list))
    print(healthy_list)

    sizes_healthy = [int(len(healthy_list)*0.5), int(len(healthy_list)*0.25), int(len(healthy_list)*0.25)]
    random.shuffle(healthy_list)
    print(healthy_list)
    it = iter(healthy_list)
    healthy_list_separated = [[next(it) for _ in range(size)] for size in sizes_healthy]
    healthy_pretext_GMM = healthy_list_separated[0]
    healthy_eval = healthy_list_separated[1]
    healthy_testset = healthy_list_separated[2]

    #healthy_pretext_list = [p for i in healthy_pretext_GMM for p in healthy['patch_name'].to_list() if i == p.split('_')[0]]
    healthy_pretext_list = [p for i in healthy_pretext_GMM for p in healthy['patch_name'].to_list() if p.startswith(i)]

    healthy_GMM_list = [p for i in healthy_pretext_GMM for p in healthy['patch_name'].to_list() if p.startswith(i) and annotation.loc[annotation['patch_name'] == p, 'emph_0.01'].iloc[0] == 0]

    #removing healthy patches that have >1% emphysema inside
    removed = [p for i in healthy_pretext_GMM for p in healthy['patch_name'].to_list() if p.startswith(i) and annotation.loc[annotation['patch_name'] == p, 'emph_0.01'].iloc[0] == 1]
    print(removed)
    df_remove = healthy[healthy['patch_name'].isin(removed)]
    df_global = healthy[healthy['patch_name'].isin(healthy_GMM_list)]
    print(df_remove)
    from collections import Counter
    print(Counter(df_global['location']))
    print(Counter(df_remove['location']))
    dict_loc_healthy = dict(Counter(df_global['location']))
    dict_loc_healthy_perc = dict(Counter(df_remove['location']))
    divide = {k: dict_loc_healthy_perc[k] / (dict_loc_healthy[k] + dict_loc_healthy_perc[k]) for k in dict_loc_healthy.keys() & dict_loc_healthy_perc}
    print(divide)

    dict_name_healthy = dict(Counter(df_global[patient]))
    dict_name_healthy_perc = dict(Counter(df_remove[patient]))
    divide = {k: dict_name_healthy_perc[k] / (dict_name_healthy[k] + dict_name_healthy_perc[k]) for k in dict_name_healthy.keys() & dict_name_healthy_perc}
    print(divide)



    #healthy_eval_list = [p for i in healthy_eval for p in healthy['patch_name'].to_list() if i == p.split('_')[0]]
    healthy_eval_list = [p for i in healthy_eval for p in healthy['patch_name'].to_list() if p.startswith(i)]

    #healthy_testset_list = [p for i in healthy_testset for p in healthy['patch_name'].to_list() if i == p.split('_')[0]]
    healthy_testset_list = [p for i in healthy_testset for p in healthy['patch_name'].to_list() if p.startswith(i)]



    #copd_list = COPD['patient'].to_list()
    copd_list = list(set(COPD[patient]))

    copd_list = [x for x in copd_list]#[x.lower() for x in copd_list]

    sizes_copd = [int(len(copd_list)*0.5), int(len(copd_list)*0.25), int(len(copd_list)*0.25)]
    random.shuffle(copd_list)
    it = iter(copd_list)
    copd_list_separated = [[next(it) for _ in range(size)] for size in sizes_copd]
    copd_pretext = copd_list_separated[0]
    copd_eval = copd_list_separated[1]
    copd_testset = copd_list_separated[2]

    #copd_pretext_list = [p for i in copd_pretext for p in COPD['patch_name'].to_list() if i == p.split('_')[0]]
    copd_pretext_list = [p for i in copd_pretext for p in COPD['patch_name'].to_list() if p.startswith(i)]

    #copd_eval_list = [p for i in copd_eval for p in COPD['patch_name'].to_list() if i == p.split('_')[0]]
    copd_eval_list = [p for i in copd_eval for p in COPD['patch_name'].to_list() if p.startswith(i)]

    #copd_testset_list = [p for i in copd_testset for p in COPD['patch_name'].to_list() if i == p.split('_')[0]]
    copd_testset_list = [p for i in copd_testset for p in COPD['patch_name'].to_list() if p.startswith(i)]

    copd_pretext_list = list(set(copd_pretext_list))
    healthy_pretext_list = list(set(healthy_pretext_list))
    healthy_GMM_list = list(set(healthy_GMM_list))
    healthy_eval_list = list(set(healthy_eval_list))
    healthy_testset_list = list(set(healthy_testset_list))
    copd_eval_list = list(set(copd_eval_list))
    copd_testset_list = list(set(copd_testset_list))
    print('final for training')
    print('num_copd_pretext', len(copd_pretext_list))
    print('num_healthy_pretextGMM', len(healthy_pretext_list))
    print('num_healthy_eval', len(healthy_eval_list))
    print('num_healthy_testset', len(healthy_testset_list))
    print('num_copd_eval', len(copd_eval_list))
    print('copd_testset', len(copd_testset_list))


    filenames_for_pretext = copd_pretext_list + healthy_pretext_list
    filenames_for_GMM = healthy_GMM_list
    filenames_for_eval = copd_eval_list + healthy_eval_list
    filenames_for_testset = copd_testset_list + healthy_testset_list
    #saves the list of patients in txt file
    with open(os.path.join(data_folder[0], 'overlap' + overlap, 'patches_for_pretext.txt'), 'w') as fp:
        json.dump(filenames_for_pretext, fp)
    with open(os.path.join(data_folder[0], 'overlap' + overlap, 'patches_for_GMM.txt'), 'w') as fp:
        json.dump(filenames_for_GMM, fp)
    with open(os.path.join(data_folder[0], 'overlap' + overlap, 'patches_for_eval.txt'), 'w') as fp:
        json.dump(filenames_for_eval, fp)
    with open(os.path.join(data_folder[0], 'overlap' + overlap, 'patches_for_testset.txt'), 'w') as fp:
        json.dump(filenames_for_testset, fp)
    with open(os.path.join(data_folder[0], 'overlap' + overlap, 'patches_copd.txt'), 'w') as fp:
        json.dump(copd_pretext_list + copd_eval_list + copd_testset_list, fp)
    with open(os.path.join(data_folder[0], 'overlap' + overlap, 'patches_healthy.txt'), 'w') as fp:
        json.dump(healthy_pretext_list + healthy_GMM_list + healthy_eval_list + healthy_testset_list, fp)


def select_patients_for_realworldscenario(data_folder, overlap, dataset, kfold):
    if dataset == 'cosyconet':
        patient = 'patient'
        fev = 'FEV1_GLI'
        fvc = 'FVC_GLI'
        gold = 'GOLD_gli'
        gold_threshold = 1
    elif dataset == 'copdgene':
        patient = 'SUBJECT_ID'
        fev = 'FEV1_post'
        fvc = 'FVC_post'
        gold = 'finalGold'
        gold_threshold = 0

    with open(os.path.join(data_folder[0], 'overlap' + overlap, 'fold' + str(kfold) + '_org', 'patches_for_pretext.txt'), "r") as fp:
        list_filenames = json.load(fp)
    patients = np.unique([i.split('_')[0] + '_' + i.split('_')[1] for i in list_filenames])

    annotation = pd.read_csv(
        os.path.join(data_folder[0].replace('/pre-processed', ''), 'all_info_patches_COPD' + overlap + '.csv'),
        sep=',', converters={patient: lambda x: str(x)})

    #annotation = annotation[annotation.notna()]
    #annotation.dropna(axis=0, how='any', inplace=True)


    ######
    annotation_drop = annotation.drop_duplicates(subset=[patient])
    healthy = annotation.loc[annotation[gold] <= gold_threshold]
    copd = annotation.loc[annotation[gold] > gold_threshold]

    # global prevelance of copd 10.3%
    copd_prevalence = 0.15 #0.103
    healthy_prevalence = 1 -copd_prevalence


    # healthy_list = healthy['patient'].to_list()
    healthy_patches_list = healthy[patient]
    healthy_list = list(set(healthy[patient]))

    healthy_list = [x for x in healthy_list]  # [x.lower() for x in healthy_list]
    # remove duplicates
    healthy_list = list(set(healthy_list))
    print(healthy_list)

    copd_list = list(set(copd[patient]))

    copd_list = [x for x in copd_list]

    healthy_pretext = [name for name in patients if name in healthy_list]
    copd_pretext = [name for name in patients if name in copd_list]
    random.seed(3)
    copd_realworld_patients = random.sample(copd_pretext, int(len(healthy_pretext)*copd_prevalence/healthy_prevalence))
    copd_realworld_patches = [p for i in copd_realworld_patients for p in list_filenames if p.startswith(i)]
    healthy_patches = [p for i in healthy_pretext for p in list_filenames if p.startswith(i)]

    final_real_world = copd_realworld_patches + healthy_patches
    print(final_real_world)
    with open(os.path.join(data_folder[0], 'overlap' + overlap, 'fold' + str(kfold), 'patches_for_realworld_pretext.txt'), 'w') as fp:
        json.dump(final_real_world, fp)



if __name__ == "__main__":
    dataset = 'copdgene'
    overlap = '20'
    fold = 5
    data_folder = ['/home/silvia/E132-Projekte/Projects/2021_Silvia_COPDGene/from_Oliver/CRADL/pre-processed']#['/home/silvia/Documents/CRADL/pre-processed/no_resample']
    #select_patients_fortasks_emphysema_withfolds(data_folder, overlap, dataset, n_folds=5)
    select_patients_for_realworldscenario(data_folder, overlap, dataset, kfold=1)