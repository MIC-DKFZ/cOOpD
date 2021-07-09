import torch


class MapDataset(torch.utils.data.Dataset):
    """
    Given a dataset, creates a dataset which applies a mapping function
    to its items (lazily, only when an item is called).

    Note that data is not cloned/copied from the initial dataset.
    """

    def __init__(self, dataset, transform=None, transform_target=None):
        self.dataset = dataset
        self.transform = transform
        self.transform_target = transform_target

    def __getitem__(self, index):
        item, target = self.dataset[index]
        if self.transform:
            item = self.transform(item)
        if self.transform_target:
            target = self.transform_target(target)
        return item, target

    def __len__(self):
        return len(self.dataset)

def get_same_index(ds, label, invert=False):
    label_indices = []
    for i in range(len(ds)):
        if invert:
            if ds[i][1] != label:
                label_indices.append(i)
        if not invert:
            if ds[i][1] == label:
                label_indices.append(i)
    return label_indices



def split_ood_label(data_set, ood_class, keep_labels=True, single_class=False):
    indices_zero = get_same_index(data_set, ood_class, invert=True)
    indices_one = get_same_index(data_set, ood_class, invert=False)

    if single_class is True:
        if keep_labels is False: 
            mask = [data_set.targets==ood_class]
            data_set.targets[mask[0]] = 0
            data_set.targets[~(mask[0])] = 1
        out_set = torch.utils.data.Subset(data_set, indices_zero)
        in_set = torch.utils.data.Subset(data_set, indices_one)

    else:
        if keep_labels is False: 
            mask = [data_set.targets!=ood_class]
            data_set.targets[mask[0]] = 0
            data_set.targets[~(mask[0])] = 1
        in_set = torch.utils.data.Subset(data_set, indices_zero)
        out_set = torch.utils.data.Subset(data_set, indices_one)

    return in_set, out_set


def get_dataset_split(train_set, val_set, test_set, sort_class, keep_labels=False, single_class=True):
    train_in, train_out = split_ood_label(train_set, ood_class=sort_class, single_class=single_class)
    val_in, val_out = split_ood_label(val_set, ood_class=sort_class, single_class=single_class)
    test_in, test_out = split_ood_label(test_set, ood_class=sort_class, single_class=single_class)

    return train_in, train_out, val_in, val_out, test_in, test_out


def to_conv(inputs, size):
    return inputs.view(-1, 1, size, size)