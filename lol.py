from tkinter import *
import random
import json
import os
import time

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

#Haragdah zagvariig beltgej bui ni
root = Tk()
root.geometry("800x500")
root.title(" ЧатБот ")
root.configure(background='light blue')

w = Label(root, text="ОЛОНЛОГЭГЗЭ СУРГУУЛЬ - МАТЕМАТИКИЙН ТОМЪЁОНЫ ЛАВЛАХ")
w.pack()  

#Mdeej data gdeg zuil bga uchraas, teriigee holboh bas bus endees ehelj bn
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
model.load_state_dict(model_state)
model.eval()


#Yag ehleh uyd gants yavagdah heseg
bot_name = "Чатбот"
ehleh=Label(root, text="Чатботыг эхлүүллээ. Хэрэв та болих бол stop гэж бичнэ үү!")

#Yeronhii heseg
def Take_input():
    sentence = inputtxt.get("1.0", "end-1c")
    Output.insert(END, 'Та : ', sentence)

    if sentence == "stop":
        exit()

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                Output.insert(END, "Хариулагч : {random.choice(intent['responses'])}")
    else:
        Output.insert(END, "Хариулагч : Ойлгомжгүй байна. Дахиад өөрөөр лавлана уу")




inputtxt = Text(root, height = 3, width = 60, bg = "light yellow", font=("Courier", 10))

Output = Text(root, height = 20, width = 98, bg = "light cyan")
  
Display = Button(root, height = 2, width = 20,  text ="Send", command = lambda:Take_input())   

zai=Label(root, text="\n", bg="light blue")




#main console
ehleh.pack()
zai.pack()
Output.pack() 
inputtxt.pack(padx=5, pady=10, side=RIGHT)
Display.pack(padx=10, pady=10, side=RIGHT)
          
  
mainloop()