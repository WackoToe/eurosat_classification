import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm
from torchmetrics import ConfusionMatrix

class Net(nn.Module):
    def __init__(self, num_classes, net_depth):
        super(Net, self).__init__()

        self.num_classes = num_classes
        self.net_depth = net_depth

        if self.net_depth == 3:
            self.fc1 = nn.Linear(2048, 1024)
            self.fc2 = nn.Linear(1024, 512)
            self.fc3 = nn.Linear(512, num_classes)
        elif self.net_depth == 2:
            self.fc1 = nn.Linear(2048, 512)
            self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        if self.net_depth == 3:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        elif self.net_depth == 2:
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
        return x


def train_loop(device, model, epochs, features_train_dataloader):
    import torch.optim as optim

    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

    # for epoch in tqdm(range(100)):  # loop over the dataset multiple times
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        print("Epoch {}/{}".format(epoch, epochs-1))
        for i, data in enumerate(features_train_dataloader, 0):
            # print(data[1].size())
            # exit()
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # print(inputs.is_cuda, labels.is_cuda)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # if i % 10 == 9:  # print every 10 mini-batches
            #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
            #     running_loss = 0.0
        print("\tLoss: {}".format(running_loss))

    print('***** Finished Training *****')
    return model


def valid_loop(device, model, features_valid_dataloader, num_classes):
    model.to(device)
    correct = 0
    total = 0
    confusion_matrix = torch.zeros((num_classes, num_classes))
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in features_valid_dataloader:
            features, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = model(features)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)

            for i in range(len(predicted)):
                confusion_matrix[predicted[i]][labels[i]] += 1

            predicted.to(device)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # confusion_matrix /= len(features_valid_dataloader.dataset)
    print(confusion_matrix)
    print(f'Accuracy of the network on the valid images: {100 * correct // total} %')

