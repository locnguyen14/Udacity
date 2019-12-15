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
    
    # Create a parser object
    parser = argparse.ArgumentParser(description="Neural Network Prediction")

    # Add argument to the parser object
    parser.add_argument('--top_k', action='store', dest='top_k', type=int, default=3, help='Number of top result')
    parser.add_argument('--category_names', action='store', dest='cat', type=str, default='cat_to_name.json', help='json name to map catgories')
    parser.add_argument('--gpu', action='store_true', dest='gpu', default=False, help='Use GPU if --gpu')
    parser.add_argument('--st', action='store_true', default='False', dest='start', help='--st to start predicting')
    parser.add_argument('--img', action = 'store', dest = 'img', type = str, default = 'Sample_image.jpg',help = 'Store Img Name')
    
    # Parse the argument from standard input
    args = parser.parse_args()

    # Print out parsing/default parameters
    print('---------Parameters----------')
    print('gpu              = {!r}'.format(args.gpu))
    print('img              = {!r}'.format(args.img))
    print('top_k            = {!r}'.format(args.top_k))
    print('cat              = {!r}'.format(args.cat))
    print('start            = {!r}'.format(args.start))

    print('-----------------------------')
    
    if args.start == True:
        model, class_labels = helper.load_saved_models()
        cat_to_name, label_order = helper.load_json(args.cat)
        ps, labels, index = helper.predict(args.img, model, args.top_k, cat_to_name, class_labels, args.gpu)
        print("------------------Prediction------------------")
        for i in range(len(ps)):
            print("The probability of the flower to be {} is {:.2f} %.".format(labels[i], ps[i] * 100))
   
if __name__ == "__main__":
    main()
    
    
    