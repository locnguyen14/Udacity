import argparse
import helper
import matplotlib.pyplot as plt
import json
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image

def main():
    # Create parser object
    parser = argparse.ArgumentParser(description="Neural Network Image Classifier Training")
    
    # Define argument for parser object
    parser.add_argument('--save_dir', type=str, help='Define save directory for checkpoints model as a string')
    parser.add_argument('--arch', dest='arch', action ='store', type = str, default = 'densenet', help='choose a tranfer learning model or architechture')
    parser.add_argument('--learning_rate', dest = 'learning_rate', action='store', type=float, default=0.001, help='Learning Rate for Optimizer')
    parser.add_argument('--hidden_units', dest = 'hidden_units', action='store', type=int, default=512, help='Define number of hidden unit')
    parser.add_argument('--epochs', dest = 'epochs', action='store', type=int, default=1, help='Number of Training Epochs')
    parser.add_argument('--gpu', dest = 'gpu', action='store_true', default = 'False', help='Use GPU if --gpu')
    parser.add_argument('--st', action = 'store_true', default = False, dest = 'start', help = '--st to start predicting')
                        
    # Parse the argument from standard input
    args = parser.parse_args()
    
    # Print out the passing/default parameters
    print('-----Parameters------')
    print('gpu              = {!r}'.format(args.gpu))
    print('epoch(s)         = {!r}'.format(args.epochs))
    print('arch             = {!r}'.format(args.arch))
    print('learning_rate    = {!r}'.format(args.learning_rate))
    print('hidden_units     = {!r}'.format(args.hidden_units))
    print('start            = {!r}'.format(args.start))
    print('----------------------')
    
    if args.start == True:
        class_labels, trainloaders, testloaders, validloaders = helper.load_image()
        model = helper.load_pretrained_model(args.arch, args.hidden_units)
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr = args.learning_rate)
        helper.train_model(model, args.learning_rate, criterion, trainloaders, validloaders, args.epochs, args.gpu)
        helper.test_model(model, testloaders, args.gpu)
        model.to('cpu')
        
        # saving checkpoints
        helper.save_checkpoint({
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'hidden_units': args.hidden_units,
            'class_labels': class_labels
        })
        print('model checkpoint saved')
                         
if __name__ == "__main__":
    main()