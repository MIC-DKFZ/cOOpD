from batchgenerators.utilities.file_and_folder_operations import *
import pandas as pd
import numpy as np



def metadata_to_DataFrame(folder):
    all_metadata = []
    for file in os.listdir(folder):
        if file.endswith(".pkl"):
            metadata = load_pickle(os.path.join(folder, file))
            all_metadata.append(metadata)
    print(all_metadata)
    df_metadata = pd.DataFrame(all_metadata)
    print(df_metadata)

    patches_names = df_metadata['patient'].to_numpy()
    print(patches_names)
    patches_names = list(set(patches_names))


    annotation = pd.read_csv(
        os.path.join(data_folder[0], 'COPD_criteria_complete.csv'),
        sep=',', converters={'patient': lambda x: str(x)})

    annotation = annotation[annotation.notna()]
    annotation = annotation.dropna(subset=["condition_COPD_GOLD"])
    FEV1_GLI = [0] * df_metadata.shape[0]
    FVC_GLI = [0] * df_metadata.shape[0]
    GOLD_gli = [0] * df_metadata.shape[0]
    GOLDclass_gli = [0] * df_metadata.shape[0]

    N_patches = [0] * df_metadata.shape[0]
    for name_csv in annotation['patient'].values:
        # for name_meta in df_metadata['patient'].values:
        #     if name_csv == name_meta:
        N_patches += np.where(df_metadata['patient'] == name_csv, 1, False)
        FEV1_GLI += np.where(df_metadata['patient'] == name_csv, annotation['FEV1_GLI'][annotation['patient']== name_csv], False)
        FVC_GLI += np.where(df_metadata['patient'] == name_csv, annotation['FVC_GLI'][annotation['patient']== name_csv], False)
        GOLD_gli += np.where(df_metadata['patient'] == name_csv, annotation['GOLD_gli'][annotation['patient']== name_csv], False)
        GOLDclass_gli += np.where(df_metadata['patient'] == name_csv, annotation['GOLDclass_gli'][annotation['patient']== name_csv], False)
    df_metadata['N_patches'] = N_patches
    df_metadata['FEV1_GLI'] = FEV1_GLI / N_patches
    df_metadata['FVC_GLI'] = FVC_GLI / N_patches
    df_metadata['GOLD_gli'] = GOLD_gli / N_patches
    df_metadata['GOLDclass_gli'] = GOLDclass_gli / N_patches
    df_metadata['FEV_FVC'] = df_metadata['FEV1_GLI']/df_metadata['FVC_GLI']
    print(df_metadata)
    df_metadata['emph_0.01'] = df_metadata['emphysema_perc'].apply(lambda x: 0 if x < 0.01 else 1)
    df_metadata['patch_name'] = df_metadata["patient"].astype(str) + '_' + df_metadata["patch_num"].astype(str)
    df_metadata.to_csv(os.path.join(data_folder[0].replace('/pre-processed', ''), 'all_info_patches_COPD.csv'))


if __name__ == "__main__":
    folder = '/home/silvia/Documents/CRADL/pre-processed/patches_new_all_overlap0'
    data_folder = ['/home/silvia/Documents/CRADL/pre-processed']

    metadata_to_DataFrame(folder)

