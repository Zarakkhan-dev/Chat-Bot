import random
import json

import torch

from model import NeuralNet
from nltk_utilis import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state) #KNow ours parameter
model.eval() #evalution paramter

bot_name = "AI Assistent"
print("Let's chat! (type 'quit' to exit)")
sentence=""
while sentence !="quit":
    # sentence = "do you use credit cards?"
    sentence = input("Enter The Promopt:  ")
    if sentence == "quit" :
        break

    sentence = tokenize(sentence)

    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0]) #reshape to get only one rows of set
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()] #Get the actual Tags 

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]#Probability for this tag is high enough and to do this 
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    
    else:
        print(f"{bot_name}: I do not understand...")




# from nltk_utilis import bag_of_words, tokenize, stem

# fro
# word = "hellu";

# print(stem(word))