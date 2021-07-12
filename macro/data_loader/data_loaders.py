from torchvision import datasets, transforms
from base import BaseDataLoader
import data_loader.datasets as Datasets
from utils.data_util import get_transform


class DataLoader(BaseDataLoader):
    def __init__(self, dataset_name, root_dir, partition, split, batch_size,
                 neg_factor=2, clip_len=16, stride=1, transform_type='sc', num_workers=8):

        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.partition = partition
        self.split = split

        self.batch_size = batch_size
        self.neg_factor = 2
        self.clip_len = clip_len
        self.stride = stride

        shuffle = True if self.partition == 'train' else False
        self.shuffle = shuffle
        # for occurred error due to limited samples and the usage of DataParallel
        if self.dataset_name == 'CASME':
            drop_last = True if self.partition == 'train' else False
        else: # 'SAMM'
            # drop_last = True if self.partition == 'train' else False # 2021/07/02, error for split '028' (22/30) of SAMM
            drop_last = True # 2021/07/02, error for split '037' (30/30) of SAMM
            # drop_last = True # note that some videos (fortunately, all belong to non-expression samples) in the end will not be used for validation
        self.drop_last = drop_last

        self.transform = get_transform(partition, type=transform_type)

        Dataset = getattr(Datasets, f"{self.dataset_name}Dataset")
        self.dataset = Dataset(root_dir=root_dir, partition=partition, split=split,
                               n_frames=clip_len, neg_factor=neg_factor, stride=stride, transform=self.transform)


        super().__init__(self.dataset, batch_size, shuffle, 0, drop_last, num_workers)