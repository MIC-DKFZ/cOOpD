import torch
import torch.nn as nn
import torch.nn.functional as F
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.abstract_transforms import Compose, RndTransform, AbstractTransform

class COPDDataloader(DataLoader):
    def __init__(self, data, batch_size, num_threads_in_multithreaded, seed_for_shuffle=1234, return_incomplete=False,
                 shuffle=True):
        """
        data must be a list of patients as returned by get_list_of_patients (and split by get_split_deterministic)
        patch_size is the spatial size the retured batch will have
        """
        super().__init__(data, batch_size, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete, shuffle,
                         True)
        self.num_modalities = 1
        self.indices = list(range(len(data['labels'])))

    def generate_train_batch(self):
        # DataLoader has its own methods for selecting what patients to use next, see its Documentation
        idx = self.get_indices()
        patients_for_batch = [self._data['patients'][i] for i in idx]

        # shape = [self._data['latent'][i].shape for i in idx][0]
        # print(shape)
        # data = np.zeros((self.batch_size, *shape), dtype=np.float32)
        # print(data.shape)


        # initialize empty array for data and seg
        labels = []
        patient_names = []

        # iterate over patients_for_batch and include them in the batch
        for i, j in enumerate(patients_for_batch):
            print(i, j)
            print(self._data['patients'].index(j))
            print(self._data['labels'][self._data['patients'].index(j)])

            patient_data, metadata = self._data['latent'][self._data['patients'].index(j)], self._data['labels'][self._data['patients'].index(j)]
            #patient_data, metadata = self.load_patient(os.path.join(self.directory, j))
            #patient_data = np.moveaxis(patient_data, 1, -1)

            data = patient_data
            labels.append(metadata)
            patient_names.append(j)

        return {'data': data, 'label': labels, 'patient': patient_names}

def reconstruct_vector_attentionmech(dict_output):
    print(dict_output)
    patient_name = dict_output['patient'].cpu().detach().numpy()
    patch_num = dict_output['patch_number'].cpu().detach().numpy()
    location = dict_output['location'].cpu().detach().numpy()
    latent = dict_output['latent'].cpu().detach().numpy()
    labels = dict_output['labels'].cpu().detach().numpy()

    p = patient_name.argsort()

    patient_name_org = patient_name[p]
    patch_num_org = patch_num[p]
    location_org = location[p]
    latent_org = latent[p]
    labels_org = labels[p]
    indexes = [index for index, _ in enumerate(patient_name_org) if
               patient_name_org[index] != patient_name_org[index - 1]]
    indexes.append(len(patient_name_org))
    final_patient_name = [patient_name_org[indexes[i]:indexes[i + 1]] for i, _ in enumerate(indexes) if
                          i != len(indexes) - 1]
    final_patch_num_org = [patch_num_org[indexes[i]:indexes[i + 1]] for i, _ in enumerate(indexes) if
                           i != len(indexes) - 1]
    final_location_org = [location_org[indexes[i]:indexes[i + 1]] for i, _ in enumerate(indexes) if
                          i != len(indexes) - 1]
    final_latent_org = [latent_org[indexes[i]:indexes[i + 1]] for i, _ in enumerate(indexes) if
                        i != len(indexes) - 1]
    final_labels_org = [labels_org[indexes[i]:indexes[i + 1]] for i, _ in enumerate(indexes) if
                        i != len(indexes) - 1]
    final_patient_name = [max(map(int, i)) for i in final_patient_name]
    final_labels_org = [max(map(int, i)) for i in final_labels_org]

    dict_out = {'latent': final_latent_org, 'labels': final_labels_org, 'patients': final_patient_name}

    return dict_out

class MixOrderTransform(AbstractTransform):
    """Shuffle order of vectors"""

    def __init__(self, data_key="data",  p_per_sample=1):
        self.p_per_sample = p_per_sample
        self.data_key = data_key

    def __call__(self, **data_dict):
        if np.random.uniform() < self.p_per_sample:
            data_dict[self.data_key] = (data_dict[self.data_key])[torch.randperm((data_dict[self.data_key]).shape[0])]
        return data_dict

def get_train_transform():
    tr_transforms = []
    tr_transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.30))
    tr_transforms.append(MixOrderTransform(p_per_sample=0.3))
    tr_transforms = Compose(tr_transforms)
    return tr_transforms

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1


        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        print(x.shape)
        A = self.attention(x)  # NxK
        print(A.shape)
        A = torch.transpose(A, 1, 0)  # KxN
        print(A.shape)
        A = F.softmax(A, dim=1)  # softmax over N
        print(A.shape)

        M = torch.mm(A, x)  # KxL
        print(M.shape)


        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        print(Y)
        print(Y_hat)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A