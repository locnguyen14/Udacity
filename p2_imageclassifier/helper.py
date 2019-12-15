import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
import json
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image

def load_image():
    '''
    A simple function to load the image from folder. Take no input, and return DataLoader object
    '''
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Transformation
    train_transforms = transforms.Compose([transforms.RandomRotation(30), 
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    testloaders = torch.utils.data.DataLoader(test_datasets, batch_size=64)
    validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=64)
    class_labels = train_datasets.classes
    
    return class_labels, trainloaders, testloaders, validloaders

def load_json(filepath):
    '''
    A function to map the numbered label to name label
    '''
    with open(filepath, 'r') as f:
        cat_to_name = json.load(f, object_pairs_hook=OrderedDict)
    label_order = [1, 10, 100, 101, 102, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 3, 30,
                   31, 32, 33, 34,35, 36, 37, 38, 39, 4, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 5,  50, 51, 52, 53, 54, 55, 56,
                   57, 58, 59, 6, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 7, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 8, 80, 81,
                   82, 83, 84, 85, 86, 87, 88, 89, 9, 90, 91, 92, 93,94, 95, 96, 97, 98, 99]
    return cat_to_name, label_order

def load_pretrained_model(model_name, hidden_units):
    '''
    A function to easily build or import the pretrained model 
        Input: specified name of the pretrained model, usually a CNN
        Output: pretrained model with our definied fully connected neural network
    '''
    if model_name == 'densenet':
        model = models.densenet161(pretrained=True)
        # Freeze the parameters first
        for param in model.parameters():
            param.requires_grad = False

        # Build your own classifier
        classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(2208,hidden_units)),
                                                      ('relu1',nn.ReLU()), 
                                                      ('dropout1', nn.Dropout(p=0.2)),
                                                      ('fc2', nn.Linear(hidden_units,500)),
                                                      ('relu2', nn.ReLU()),
                                                      ('dropout2', nn.Dropout(p=0.2)),
                                                      ('fc3', nn.Linear(500,102)),
                                                      ('output', nn.LogSoftmax(dim=1))]))
    else:
        model = models.vgg13(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088,hidden_units)),
                                                      ('relu1',nn.ReLU()), 
                                                      ('dropout1', nn.Dropout(p=0.2)),
                                                      ('fc2', nn.Linear(hidden_units,500)),
                                                      ('relu2', nn.ReLU()),
                                                      ('dropout2', nn.Dropout(p=0.2)),
                                                      ('fc3', nn.Linear(500,102)),
                                                      ('output', nn.LogSoftmax(dim=1))]))
    model.classifier = classifier
    return model
        
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    

def load_saved_models(filename='checkpoint.pth.tar'):
    '''
    A function for to load the checkpoint of a saved trained model
        Input: the checkpoint file
        Output: the checkpoint model and class labels
    Notes: also remember to load the (untrain) pretrained model first and also define the optimizer for eveything to work
    '''
    checkpoint = torch.load(filename)
    model = load_pretrained_model(checkpoint['arch'], checkpoint['hidden_units'])
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = optim.Adam(model.classifier.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, checkpoint['class_labels']
    
def train_model(model, learning_rate, criterion, trainloader, validloader, epochs, gpu=False):
    '''
    A function to start training the neural network model
        Input: a neccessary parameters (models, learning rate, criterion, trainloader, validloader, epochs, gpu default)
        Output: A trained neural network model with result from validation set
    '''
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    epochs = epochs
    print_every = 40
    steps = 0
    
    # Switch to GPU mode if it's on
    if gpu is True:
        model.to('cuda')
    
    # Return the model if number of epochs is 0
    if epochs == 0:
        return
    
    # Start training the model
    for e in range(epochs):
        total_loss = 0
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            
            # Move input and model into GPU
            if gpu is True:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
            # Training process: zero_grad + forward + backward + optimize
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Print Statistics
            running_loss += loss.item()
            total_loss += loss.item()
            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs),
                    "Loss every 40 steps: {:.4f}".format(running_loss/print_every))
                running_loss = 0  
       
        # Validation part within the training set
        # Turn on evaluation mode
        model.eval()

        # Turn off gradients updating for inference part
        with torch.no_grad():
            train_loss, train_accuracy = validation(model, trainloader, criterion, gpu)
            valid_loss, valid_accuracy = validation(model, validloader, criterion, gpu)
            print("Epoch:{}/{}".format(e+1, epochs),
                  "Training Loss: {:.3f}..".format(train_loss),
                  "Trainning Accuracy: {:.3f}..".format(train_accuracy),
                  "Validation Loss: {:.3f}..".format(valid_loss),
                  "Valiation Accuracy: {:.3f}".format(valid_accuracy))

        # Get back to training after inference
        total_loss = 0
        model.train()

def test_model(model, testloader, gpu = False):
    correct = 0
    total = 0
    if gpu is True:
        model.to('cuda')
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if gpu is True:
                images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model.forward(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()
    print('Accuracy of the network on the test images: %.2f %%' % (100 * correct / total))
    
    
def validation(model, valid_loader, criterion, gpu=False):
    '''
    A validation function to check for the accuracy in validation and test set after 1 epocch
        Input: training model, validation/test loader
        Output: accuracy of the model on the validation/test loader
    '''
    correct = 0
    total = 0
    accuracy = 0
    test_loss = 0
    for inputs, labels in valid_loader:
        if gpu is True:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
        outputs = model.forward(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        test_loss += criterion(outputs, labels).item()
    accuracy = 100*correct/total
    return test_loss, accuracy
    
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Defining all transformation steps
    img_loader = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # Load the image from PIL and apply the transformation
    pil_img = Image.open(image)
    pil_img = img_loader(pil_img).float()
    
    # Convert to numpy array
    np_image = np.array(pil_img)
    
    return np_image


def predict(image_path, model, topk, cat_to_name, class_labels, gpu = False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Turn on evaluation mode, load the model and image into CPU
    model.eval()
    image = process_image(image_path)
    torch_image = torch.from_numpy(image).type(torch.FloatTensor)
    torch_image.resize_([1, 3, 224, 224])
    
    # Make the prediction in log and then convert to linear scale
    model.to("cpu")
    if gpu is True:
        print('Using GPU to Predict')
        model.to('cuda')
        torch_image = torch_image.to('cuda')
    result = torch.exp(model(torch_image))
    
    # Get top k results
    ps, index = result.topk(topk)
    
    # Detach all of the details
    ps, index = ps.detach(), index.detach()
    
    #  Attach seperate probability, index, and label list
    ps.resize_([topk])
    index.resize_([topk])
    ps, index = ps.tolist(), index.tolist()
    
    # Get the class labels
    label_index = []
    for i in index:
        label_index.append(int(class_labels[int(i)]))
    labels = []
    for i in label_index:
        labels.append(cat_to_name[str(i)])
    return ps, labels, label_index
    
def imshow(image, ax=None, title=None, normalize=True):
    ''' imshow function for a tensor'''
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1) # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    
    ax.imshow(image)
    return ax

