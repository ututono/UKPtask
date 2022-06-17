import torch
from torch import nn
import time
import torchtext
import numpy as np
import pandas as pd
import torch.optim as optim
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict, Counter
import os
import sys
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import io
from collections import Counter
from typing import List, Tuple

from LSTMClassifier import LSTMClassifier
from Reader import Reader,EmbeddingsReader
from Trainer import Trainer,Evaluator

# %config InlineBackend.figure_format = 'retina'
plt.style.use('seaborn')



def fit(model, labels, optimizer, loss, epochs, batch_size, train, valid, test):

    trainer = Trainer(optimizer)
    evaluator = Evaluator()
    best_acc = 0.0

    for epoch in range(epochs):
        print('EPOCH {}'.format(epoch + 1))
        print('=================================')
        print('Training Results')
        cm = trainer.run(model, labels=labels, train=train, loss=loss, batch_size=batch_size)
        print('Validation Results')
        cm = evaluator.run(model,valid,labels=labels)
        print(cm.get_all_metrics())
        if cm.get_acc() > best_acc:
            print('New best model {:.2f}'.format(cm.get_acc()))
            best_acc = cm.get_acc()
            torch.save(model.state_dict(), './checkpoint.pth')
    if test:
        model.load_state_dict(torch.load('./checkpoint.pth'))
        cm = evaluator.run(model,valid,labels=labels)
        print('Final result')
        print(cm.get_all_metrics())
    return cm.get_acc()

if __name__ == '__main__':
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"

    BASE = 'data'
    TRAIN = os.path.join(BASE, 'train.conll')
    VALID = os.path.join(BASE, 'dev.conll')
    TEST = os.path.join(BASE, 'test.conll')
    PRETRAINED_EMBEDDINGS_FILE = 'word_embedding/glove.6B.50d/glove.6B.50d.txt'

    # load data from the file and put them into torch.dataset
    r = Reader((TRAIN, VALID, TEST))
    train = r.load(TRAIN)
    valid = r.load(VALID)
    test = r.load(TEST)

    # load pre-trained embedding vectors
    embeddings, embed_dim = EmbeddingsReader.from_text(PRETRAINED_EMBEDDINGS_FILE, r.vocab)

    # map the NEG tag to the number
    label2index = {l: i + 1 for i, l in enumerate(r.labels)}
    label2index['[PAD]'] = 0
    label2index[START_TAG] = len(r.labels) + 1
    label2index[STOP_TAG] = len(r.labels) + 2

    # init the LSTM classifer
    model = LSTMClassifier(embeddings, len(r.labels), embed_dim, 100, tag_to_ix=label2index, batch_size=1,
                           hidden_units=[100])

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params} parameters")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    loss = torch.nn.CrossEntropyLoss()
    loss = loss.to(device)

    learnable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(learnable_params, lr=0.001)

    # slice the train dataset in order to reduce train time while developing. You can use the full
    # training dataset to obtain the maximum performance.

    # train_mini_set = train_dataset
    train_mini_set = train[:900]
    train_mini_set = TensorDataset(train_mini_set[0], train_mini_set[1], train_mini_set[2])

    fit(model, label2index.keys(), optimizer, loss, 20, 1, train_mini_set, valid, test)



