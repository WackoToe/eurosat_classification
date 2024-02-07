import torch
import torchvision
import argparse
from termcolor import colored

import data_loader, extract_features, classifier




def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    es_tr_dl, es_va_dl = data_loader.load_dataset()
    print("Train set size: {}\nValid set size: {}".format(len(es_tr_dl.dataset), len(es_va_dl.dataset)))
    print(colored("load_dataset: ", "magenta", attrs=["bold"]) + colored("DONE", "green", attrs=["bold"]), flush=True)

    features_train_dataloader, features_valid_dataloader, num_classes = extract_features.extract_features(es_tr_dl, es_va_dl, device)
    print(colored("extract_features: ", "magenta", attrs=["bold"]) + colored("DONE", "green", attrs=["bold"]), flush=True)

    simple_net = classifier.Net(num_classes, args.net_depth)
    trained_model = classifier.train_loop(device, simple_net, args.epochs, features_train_dataloader)
    classifier.valid_loop(device, trained_model, features_valid_dataloader, num_classes)
    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, help="Number of training epochs", default=100)
    parser.add_argument("--net_depth", type=int, default=3)
    args = parser.parse_args()
    main(args)