from batchgenerators.utilities.file_and_folder_operations import *
import pandas as pd
import numpy as np



def metadata_to_DataFrame(folder, overlap, dataset):
    if dataset == 'cosyconet':
        patient = 'patient'
        fev = 'FEV1_GLI'
        fvc = 'FVC_GLI'
        gold = 'GOLD_gli'
    elif dataset == 'copdgene':
        patient = 'SUBJECT_ID'
        fev = 'FEV1_post'
        fvc = 'FVC_post'
        gold = 'finalGold'
    all_metadata = []
    for file in os.listdir(folder):
        if file.endswith(".pkl"):
            metadata = load_pickle(os.path.join(folder, file))
            all_metadata.append(metadata)
    print(all_metadata)
    df_metadata = pd.DataFrame(all_metadata)
    if dataset=='copdgene':
        df_metadata.rename(columns={'0010|0010': 'SUBJECT_ID'}, inplace=True)
    print(df_metadata)
    df_metadata[patient] = df_metadata[patient].str.strip()

    patches_names = df_metadata[patient].to_numpy()
    print(patches_names)
    patches_names = list(set(patches_names))


    annotation = pd.read_csv(
        os.path.join(data_folder[0], 'COPD_criteria_complete.csv'),
        sep=',', converters={patient: lambda x: str(x)})

    #annotation = annotation[annotation.notna()]
    annotation = annotation.dropna(subset=["condition_COPD_GOLD"])
    FEV1_GLI = [0] * df_metadata.shape[0]
    FVC_GLI = [0] * df_metadata.shape[0]
    GOLD_gli = [0] * df_metadata.shape[0]

    #GOLDclass_gli = [0] * df_metadata.shape[0]



    N_patches = [0] * df_metadata.shape[0]
    for name_csv in annotation[patient].values:
        # for name_meta in df_metadata['patient'].values:
        #     if name_csv == name_meta:
        # N_patches += np.where(df_metadata[patient] == name_csv, 1, False)
        # FEV1_GLI += np.where(df_metadata[patient] == name_csv, annotation[fev][annotation[patient]== name_csv], False)
        # FVC_GLI += np.where(df_metadata[patient] == name_csv, annotation[fvc][annotation[patient]== name_csv], False)
        # GOLD_gli += np.where(df_metadata[patient] == name_csv, annotation[gold][annotation[patient]== name_csv], False)
        #GOLDclass_gli += np.where(df_metadata[patient] == name_csv, annotation['GOLDclass_gli'][annotation[patient]== name_csv], False)
        N_patches += np.where(df_metadata[patient].str.contains(name_csv), 1, False)
        FEV1_GLI += np.where(df_metadata[patient].str.contains(name_csv), annotation[fev][annotation[patient].str.contains(name_csv)], False)
        FVC_GLI += np.where(df_metadata[patient].str.contains(name_csv), annotation[fvc][annotation[patient].str.contains(name_csv)], False)
        GOLD_gli += np.where(df_metadata[patient].str.contains(name_csv), annotation[gold][annotation[patient].str.contains(name_csv)], False)
    df_metadata['N_patches'] = N_patches
    df_metadata[fev] = FEV1_GLI / N_patches
    df_metadata[fvc] = FVC_GLI / N_patches
    df_metadata[gold] = GOLD_gli / N_patches
    #df_metadata['GOLDclass_gli'] = GOLDclass_gli / N_patches
    df_metadata['FEV_FVC'] = df_metadata[fev]/df_metadata[fvc]
    print(df_metadata)
    df_metadata['emph_0.01'] = df_metadata['emphysema_perc'].apply(lambda x: 0 if x < 0.01 else 1)
    df_metadata['patch_name'] = df_metadata[patient].astype(str) + '_' + df_metadata["patch_num"].astype(str)
    df_metadata.to_csv(os.path.join(data_folder[0].replace('/pre-processed', ''), 'all_info_patches_COPD' + overlap + '.csv'))


if __name__ == "__main__":
    dataset = 'copdgene'
    overlap = '20' #'20'

    if dataset == 'cosyconet':
        folder = '/home/silvia/Documents/CRADL/pre-processed/patches_new_all_overlap' + overlap
        data_folder = ['/home/silvia/Documents/CRADL/pre-processed']
    elif dataset == 'copdgene':
        folder = '/home/silvia/E132-Projekte/Projects/2021_Silvia_COPDGene/from_Oliver/CRADL/pre-processed/patches_new_all_overlap' + overlap
        data_folder = ['/home/silvia/E132-Projekte/Projects/2021_Silvia_COPDGene/from_Oliver/CRADL/pre-processed']

    metadata_to_DataFrame(folder, overlap, dataset=dataset)

