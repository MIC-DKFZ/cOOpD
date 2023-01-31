
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
import os
import pandas as pd
import random
directory = "/home/silvia/E132-Projekte/Projects/2020_Silvia_komorbidom/CRADL/pre-processed/cosyconet/cosyconet/"
with open(os.path.join(directory, 'overlap' + str(20), 'patches_copd.txt'), "r") as fp:
    list_filenames_copd = json.load(fp)

with open(os.path.join(directory, 'overlap' + str(20), 'patches_healthy.txt'), "r") as fp:
    list_filenames_healthy = json.load(fp)

list_filenames = list_filenames_copd+list_filenames_healthy

with open(os.path.join(directory, 'overlap' + str(20), 'fold' + str(1), 'patches_for_testset.txt'), "w") as fp:
    json.dump(list_filenames_copd+list_filenames_healthy, fp)





directory = "/dkfz/cluster/gpu/data/OE0441/s280a/CRADL/pre-processed/copdgene/"
directory = "/home/silvia/E132-Projekte/Projects/2021_Silvia_COPDGene/from_Oliver/CRADL/pre-processed/"
with open(os.path.join(directory, 'overlap' + str(0), 'fold' + str(1) + '_reduced', 'patches_for_pretext_max100.txt'), "r") as fp:
    list_filenames = json.load(fp)
#prefixes = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P' 'O', 'P', 'Q', 'S', 'T', 'U', 'V', 'W', 'Y', 'X', 'Z')
#prefixes = ('O', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', 'X', 'Z')
#prefixes = ('S', 'O') #O, Q, prob R
prefixes = ('A', 'B', 'C', 'D', 'E', 'F')
newlist = [x for x in list_filenames if x.split('_')[1].startswith(prefixes)]
with open(os.path.join(directory, 'overlap' + str(0), 'fold' + str(1) + '_reduced', 'patches_for_pretext_max100_A_B_C.txt'), "w") as fp:
    json.dump(newlist, fp)
test_list = [[x.split('_')[0] + '_' + x.split('_')[1] for x in list_filenames], list_filenames]
df_select = pd.DataFrame(test_list, index=['patient', 'full_name']).T
df1 = df_select.groupby('patient')['full_name'].apply(list).reset_index(name='new')
list_filenames_selected = []
random.seed(1)
for list_name in df1['new'].tolist():
    if len(list_name) < 100:
        list_filenames_selected.extend(list_name)
    else:
        random_listname = random.sample(list_name, 100)
        list_filenames_selected.extend(random_listname)

with open(os.path.join(directory, 'overlap' + str(20), 'fold' + str(1) + '_org', 'patches_for_pretext_reduced.txt'), 'w') as fp:
    json.dump(list_filenames_selected, fp)



not_found = []
for file in list_filenames:
    target_name = os.path.join(directory, 'patches_new_all_overlap0', file.lower() + '.npz')
    #print(target_name)
    try:
        numpy_array = np.load(target_name, mmap_mode="r")
        metadata = load_pickle(target_name.replace('.npz', '.pkl'))
        try:
            numpy_array_insp = numpy_array['insp'].astype(float)
            numpy_array_exp = numpy_array['exp'].astype(float)
            if len(numpy_array_insp.shape) != 3:
                print(numpy_array_insp.shape)
                print(target_name)

        except:
            print('error insp or exp')
            print(target_name)
            not_found.append(file)
    except:
        print('error')
        print(target_name)
        not_found.append(file)
    # print(target_name)
not_found.sort()
with open(os.path.join(directory, "file.txt"), "w") as output:
    output.write(str(not_found))