import os
import logging
import torch
import torchvision.datasets as datasets
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from utils.regime import Regime
from utils.dataset import IndexedFileDataset
from preprocess import get_transform
#from threaded_dataloader import ThreadedDataLoader
from forkonce_dataloader import ForkOnceDataLoader

def get_dataset(config, name, split='train', transform=None,
                target_transform=None, download=True, datasets_path='~/Datasets'):
    train = (split == 'train')
    root = os.path.join(os.path.expanduser(datasets_path), name)
    if name == 'cifar10':
        return datasets.CIFAR10(root=root,
                                train=train,
                                transform=transform,
                                target_transform=target_transform,
                                download=download)
    elif name == 'cifar100':
        return datasets.CIFAR100(root=root,
                                 train=train,
                                 transform=transform,
                                 target_transform=target_transform,
                                 download=download)
    elif name == 'mnist':
        return datasets.MNIST(root=root,
                              train=train,
                              transform=transform,
                              target_transform=target_transform,
                              download=download)
    elif name == 'stl10':
        return datasets.STL10(root=root,
                              split=split,
                              transform=transform,
                              target_transform=target_transform,
                              download=download)
    elif name == 'imagenet':
        if train:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'val')
        return datasets.ImageFolder(root=root,
                                    transform=transform,
                                    target_transform=target_transform)
    elif name == 'imagenet_tar':
        if train:
            root = os.path.join(root, 'imagenet_train.tar')
        else:
            root = os.path.join(root, 'imagenet_validation.tar')
        return IndexedFileDataset(root, extract_target_fn=(
            lambda fname: fname.split('/')[0]),
            transform=transform,
            target_transform=target_transform,
            **config)


_DATA_ARGS = {'name', 'split', 'transform',
              'target_transform', 'download', 'datasets_path'}
_DATALOADER_ARGS = {'batch_size', 'shuffle', 'sampler', 'batch_sampler',
                    'num_workers', 'collate_fn', 'pin_memory', 'drop_last',
                    'timeout', 'worker_init_fn'}
_TRANSFORM_ARGS = {'transform_name', 'input_size', 'scale_size', 'normalize', 'augment',
                   'cutout', 'duplicates', 'num_crops', 'autoaugment'}
_OTHER_ARGS = {'distributed', 'dist_backend', 'rank', 'world_size'}


class DataRegime(object):
    def __init__(self, regime, defaults={}, other_config={}):
        self.regime = Regime(regime, defaults)
        self.epoch = 0
        self.steps = None
        self.config = other_config
        self.get_loader(True)

    def get_setting(self):
        setting = self.regime.setting
        loader_setting = {k: v for k,
                          v in setting.items() if k in _DATALOADER_ARGS}
        data_setting = {k: v for k, v in setting.items() if k in _DATA_ARGS}
        transform_setting = {
            k: v for k, v in setting.items() if k in _TRANSFORM_ARGS}
        other_setting = {k: v for k, v in setting.items() if k in _OTHER_ARGS}
        transform_setting.setdefault('transform_name', data_setting['name'])
        return {'data': data_setting, 'loader': loader_setting,
                'transform': transform_setting, 'other': other_setting}

    def get(self, key, default=None):
        return self.regime.setting.get(key, default)

    def get_loader(self, force_update=False):
        if force_update or self.regime.update(self.epoch, self.steps):
            setting = self.get_setting()
            self._transform = get_transform(**setting['transform'])
            setting['data'].setdefault('transform', self._transform)
            self._data = get_dataset(self.config, **setting['data'])
            if setting['other'].get('distributed', False):
                if setting['other'].get('dist_backend', None) == 'hvd':
                    setting['loader']['sampler'] = DistributedSampler(
                        self._data, num_replicas=setting['other'].get('world_size', 1), 
                        rank=setting['other'].get('rank', 0))
                else:
                    setting['loader']['sampler'] = DistributedSampler(self._data)
                setting['loader']['shuffle'] = None
                # pin-memory currently broken for distributed
                if setting['loader']['num_workers'] != 0:
                    setting['loader']['pin_memory'] = False
            self._sampler = setting['loader'].get('sampler', None)

            logging.info('Initializing loader')
            #self._loader = torch.utils.data.DataLoader( 
            #self._loader = ThreadedDataLoader(
            self._loader = ForkOnceDataLoader(
                self._data, **setting['loader'])
        return self._loader

    def set_epoch(self, epoch):
        self.epoch = epoch
        if self._sampler is not None and hasattr(self._sampler, 'set_epoch'):
            self._sampler.set_epoch(epoch)

    def __len__(self):
        return len(self._data)
