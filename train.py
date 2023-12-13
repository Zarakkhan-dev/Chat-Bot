import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utilis import bag_of_words, tokenize, stem
from model import NeuralNet

with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))

# stem and lower each word
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))


# create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters 
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    if epoch == 1000:
        break;

print(f'final loss: {loss.item():.4f}')

#Save the data and save different things 

data ={
    "model_state":model.state_dict(),
    "input_size" : input_size,
    "output_size": output_size,
    "hidden_size" : hidden_size,
    "all_words" : all_words,
    "tags" :tags
}

FILE = "data.pth"
torch.save(data,FILE)#This will serialize it and save it to a pickled file 

print(f'Training complete, File saved to {FILE}')
# import json
# from nltk_utilis import tokenize,stem,bag_of_words
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from model import NeutralNetwork
# # open the file of json with read mode
# with open("intents.json",'r') as f:
#     intents = json.load(f)

# all_words = []

# tags=[]
# xy = []

# # print("solution is " , intents['intents'])
# for intent in intents["intents"]:
#     tag =intent['tag']
#     tags.append(tag)
#     for pattern in intent["patterns"]:
#         tokenize_data=tokenize(pattern)
#         #we cannot put array into array so we use extend to add arrary into array
#         all_words.extend(tokenize_data)
#         xy.append((tokenize_data,tag))

# ignore_words = ["?","!","." ,","];

# all_words = [stem(w) for w in all_words if w not in ignore_words]
# # Then We Apply The List comphrehensions

# # Unique word so use set keyword to remove the duplicate word from the list
# all_words =sorted(set(all_words))
# tags = sorted(set(tags))

# # Now Create a bag_word and take a list our x_train data

# #x_train we put all the bag of words 
# x_train=[]

# #y_train be the tags or the  assciated number for each tag
# y_train=[]

# for (pattern_sentence ,tag) in xy:
#     bag = bag_of_words(pattern_sentence,all_words)
#     x_train.append(bag)

# #  We have to tack the delievery then this will fgive us label 0 and then we have numbers of our labels

#     label = tags.index(tag)
#     y_train.append(label) # CrossEntropyLoss


# # convert into numpy

# import numpy as np

# x_train = np.array(x_train)
# y_train= np.array(y_train)

# class Chat_Data(Dataset):
#     def __init__(self) :
#         self.no_sample =len(x_train)
#         self.x_data = x_train
#         self.y_data =y_train
#     def __getitem__(self, index) :
#         return self.x_data[index] ,self.y_data[index]
#     def __len__(self):
#         return self.no_sample

# #Hyper Parameter
# batch_size=8    
# dataset =Chat_Data()
# train_loader = DataLoader(dataset ,batch_size=batch_size,shuffle=True,num_workers=1,timeout=60) # Number of worker mean processing
# # print(train_loader)
# input_Layer =len(x_train[0]);
# hidden_layer = 8
# num_classes_Layer = len(tags)
# device = torch.device('cuda'  if torch.cuda.is_available() else 'cpu')

# model = NeutralNetwork(input_Layer , hidden_layer , num_classes_Layer).to(device)

# # #Loss and Optimizer
# learning_rate = 0.001
# num_epochs = 1000
# criterion =nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters() ,lr =learning_rate)

# for epoch in range(num_epochs):
   
#     for (words,labels) in train_loader:
#         words = words.to(device)
#         labels = labels.to(dtype=torch.long).to(device)

#          #forword pass
#         outputs = model(words)
#         #it will get the predicted output and actua labels
#         loss = criterion(outputs ,labels)

#         #backward and Optimizer Step
#         #For Backward first empty the gradient 

#         optimizer.zero_grad()
#         #Backpropogation
#         loss.backward()
#         optimizer.step()
#     if (epoch+1) % 100 == 0:
#         print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# print(f'final loss: {loss.item():.4f}')
