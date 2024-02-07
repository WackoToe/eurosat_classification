import torch
import torchvision
import torch.nn as nn
from torchvision.models import resnet50


def extraction(device, model, dataloader):
    # Image transforms
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset_features = torch.empty((0, 2048))
    dataset_labels = torch.empty((0), dtype=torch.long)
    # dataset_features.to(device)
    for batch, labels in dataloader:
        # Create a PyTorch tensor with the transformed image
        t_batch = transforms(batch)

        with torch.no_grad():  # <-- no_grad context
            my_embedding = model(t_batch.to(device)).cpu()

        dataset_features = torch.cat((dataset_features, my_embedding))
        dataset_labels = torch.cat((dataset_labels, labels))

    num_classes = torch.unique(dataset_labels).size(dim=0)
    # print(torch.unique(dataset_labels))
    features_dataset = torch.utils.data.TensorDataset(dataset_features, dataset_labels)
    features_dataloader = torch.utils.data.DataLoader(features_dataset,
                                                      batch_size=32,
                                                      shuffle=True, )

    return features_dataloader, num_classes


def extract_features(train_dataloader, valid_dataloader, device):
    rn50 = resnet50(weights="IMAGENET1K_V2")
    # rn50.avgpool = nn.Identity()
    rn50.fc = nn.Identity()
    rn50.to(device)

    # Set model to evaluation mode
    rn50.eval()

    features_train_dataloader, num_classes = extraction(device, rn50, train_dataloader)
    features_valid_dataloader, num_classes = extraction(device, rn50, valid_dataloader)

    return features_train_dataloader, features_valid_dataloader, num_classes
