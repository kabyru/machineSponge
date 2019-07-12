"""
machineSponge, a SpongeBob episode-generating Recursive Neural Network Program.
Developed by Kaleb Byrum (@kabyru)
Derived from MortyFire by Sarthak Mittal (@naiveHobo)
Main file (main.py)
"""


import os

from tkinter import Tk
from tkinter.filedialog import askopenfilename

os.system("pip install numpy")
os.system("pip install gensim")
os.system("pip3 install https://download.pytorch.org/whl/cpu/torch-1.1.0-cp37-cp37m-win_amd64.whl")
os.system("pip3 install https://download.pytorch.org/whl/cpu/torchvision-0.3.0-cp37-cp37m-win_amd64.whl")
os.system("pip install tensorboardX")

import numpy as np
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from dataGenerator import SpongeData
from textVocabulary import Vocabulary
from modelGenerate import MachineSponge
from textGenerate import generate

ap = ArgumentParser()

ap.add_argument("--mode", required=True, type=str, help="train|generate",
                choices=["train", "generate"])
ap.add_argument("--vocab_path", default=None, type=str, help="path to load vocabulary from")
ap.add_argument("--model_path", default='machinesponge.model', type=str, help="path to load trained model from")
ap.add_argument("--checkpoint_dir", default='checkpoints/', type=str, help="path to save checkpoints")
ap.add_argument("--script_len", default=200, type=int, help="length of script")
ap.add_argument("--temperature", default=1.0, type=float, help="diversity in script generated")
ap.add_argument("--start", default='SpongeBob', type=str, help="starting word of script")

args = ap.parse_args()

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('No GPU found. Training may take longer depending on hyperparameters.')

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
data_path = askopenfilename(filetypes=[("Text Files", "*.txt")], title='Select the file containing the training data.') # show an "Open" dialog box and return the path to the selected file
print(data_path)

os.system("cls")

print('Welcome to machineSponge! This program will attempt to generate an original yet familiar SpongeBob episode by using machine learning upon existing SpongeBob episode transcripts.\n\n')
print('Enter the following hyperparameters to begin the program.')
print('NOTE: Be sure to use the same hyperparameters for both training and generating!\n')

epochs = int(input("Enter the number of Epochs (trials) to be executed. (INTEGER) (Try a range of 4 to 10, will affect overall runtime): "))
batch_size = int(input("Enter the batch size for each Epoch. (INTEGER) (Try a range of 128 to 256, will affect RAM usage): "))
lstm_size = int(input("Enter the LSTM size for each batch. (INTEGER) (Try a range of 128 to 256, will affect RAM usage): "))
seq_length = int(input("Enter the length of one RNN training sequence. (INTEGER) (Try a range of 20 to 64, will overall RAM usage & runtime): "))
num_layers = int(input("Enter the number of LSTM layers to be used. (INTEGER) (Keep around 2, any above will require extensive computing power): "))
bidirectional = True
embeddings_size = int(input("Enter the size of the embeddings for Vocabulary words. (INTEGER) (Keep around 300, will ensure coherent text generation): "))
dropout = float(input("Enter the desired Dropout Rate (neuron drop) for the run. (FLOAT) (Keep around 0.5): "))
learning_rate = float(input("Enter the desired initial learning rate for the RNN. (FLOAT) (Try around 0.001, ensures coherent results): "))

with open(data_path, 'r', encoding="utf8") as f:
    text = f.read()

vocab = Vocabulary()

if args.vocab_path is None:
    vocab.add_text(text)
    vocab.save('data/vocab.pkl')
else:
    vocab.load(args.load_vocab)

print(vocab)

model = MachineSponge(vocab_size=len(vocab), lstm_size=lstm_size, embed_size=embeddings_size, seq_length=seq_length,
                  num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, train_on_gpu=train_on_gpu)

if train_on_gpu:
    model.cuda()

print(model)

if args.mode == "train":

    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    dataset = SpongeData(text=text, seq_length=seq_length, vocab=vocab)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    writer = SummaryWriter()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    losses = []
    batch_losses = []
    global_step = 0

    os.system('cls')

    print("\nInitializing training...")
    print("Hyperparameters used:")
    print("Epochs: {}".format(epochs))
    print("Batch Size: {}".format(batch_size))
    print("LSTM Size: {}".format(lstm_size))
    print("Training Sequence Length: {}".format(seq_length))
    print("Number of LSTM Layers: {}".format(num_layers))
    print("Number of Vocabulary Embeddings: {}".format(embeddings_size))
    print("Neuron Dropout Rate: {}".format(dropout))
    print("Initial Learning Rate: {}".format(learning_rate))
    for epoch in range(1, epochs + 1):
        print("Epoch: {:>4}/{:<4}".format(epoch, epochs))

        model.train()
        hidden = model.init_hidden(batch_size)

        for batch, (inputs, labels) in enumerate(data_loader):

            labels = labels.reshape(-1)

            if labels.size()[0] != batch_size:
                break

            h = tuple([each.data for each in hidden])
            model.zero_grad()

            if train_on_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()

            output, h = model(inputs, h)

            loss = criterion(output, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            hidden = h

            losses.append(loss.item())
            batch_losses.append(loss.item())

            if batch % 10 == 0:
                print("step [{}/{}]\t loss: {:4f}".format(batch, len(dataset) // batch_size, np.average(batch_losses)))
                writer.add_scalar('loss', loss, global_step)
                batch_losses = []

            global_step += 1

        print("\n----- Generating text -----")
        for temperature in [0.2, 0.5, 1.0]:
            print('----- Temperature: {} -----'.format(temperature))
            print(generate(model, start_seq=args.start, vocab=vocab, temperature=temperature, length=100))
            print()

        torch.save(model.state_dict(),
                   os.path.join(args.checkpoint_dir, "machinesponge-{}-{:04f}.model".format(epoch, np.average(losses))))
        epoch_losses = []

    writer.close()

    print("\nSaving model [{}]".format(args.model_path))
    torch.save(model.state_dict(), args.model_path)

else:
    os.system('cls')
    print("\nLoading model from [{}]".format(args.model_path))
    model.load_state_dict(torch.load(args.model_path))
    script = generate(model, start_seq=args.start, vocab=vocab, temperature=args.temperature, length=args.script_len)
    print()
    print('----- Temperature: {} -----'.format(args.temperature))
    print(script)
    print()
