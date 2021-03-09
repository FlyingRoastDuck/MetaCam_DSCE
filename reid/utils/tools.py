from .data import transforms as T
from .data.sampler import RandomMultipleGallerySampler
from .data.preprocessor import Preprocessor
from .data import IterLoader
from .tsne import showTSNE
from torch.utils.data import DataLoader


def get_vanilla_loader(datasetRoot, height, width, batch_size, workers,
                       num_instances, iters, trainset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5), T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(), normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])
    train_set = sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=datasetRoot, transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)
    train_loader.new_epoch()
    return train_loader


def get_train_loader(datasetRoot, height, width, batch_size, workers,
                     num_instances, iters, trainset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10), T.RandomCrop((height, width)),
        T.ToTensor(), normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(trainset, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
        DataLoader(Preprocessor(trainset, root=datasetRoot, transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)
    train_loader.new_epoch()
    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def get_metatrain_loader(datasetRoot, height, width, batch_size, workers,
                         num_instances, iters, trainset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])
    train_loader = []

    for ii in range(len(trainset)):
        tempSet = trainset[ii]
        train_set = sorted(tempSet)
        rmgs_flag = num_instances > 0
        if rmgs_flag:
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
        else:
            sampler = None
        train_loader.append(
            IterLoader(
                DataLoader(Preprocessor(train_set, root=datasetRoot, transform=train_transformer),
                           batch_size=batch_size, num_workers=workers, sampler=sampler,
                           shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)
        )
        train_loader[-1].new_epoch()

    return train_loader


def get_plot_loader(dataset, height, width, batch_size, workers, test_set):
    """
    for visualization
    :param dataset:
    :param height:
    :param width:
    :param batch_size:
    :param workers:
    :param test_set:
    :return:
    """
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(), normalizer
    ])

    test_loader = DataLoader(
        Preprocessor(test_set, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader
