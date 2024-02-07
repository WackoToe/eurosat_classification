import torch
import torchvision


def load_dataset():
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    load_workers = 4

    # Image transforms
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
    ])

    eurosat_train_val_data = torchvision.datasets.EuroSAT(root="/data/DATASETS/pytorch_datasets",
                                                transform = transforms,
                                                download=True)
    train_size = int(0.8 * len(eurosat_train_val_data))
    val_size = len(eurosat_train_val_data) - train_size
    eurosat_train_data, eurosat_val_data = torch.utils.data.random_split(eurosat_train_val_data, [train_size, val_size])
    eurosat_train_dataloader = torch.utils.data.DataLoader(eurosat_train_data,
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=load_workers)
    eurosat_val_dataloader = torch.utils.data.DataLoader(eurosat_val_data,
                                                           batch_size=4,
                                                           shuffle=True,
                                                           num_workers=load_workers)
    return eurosat_train_dataloader, eurosat_val_dataloader