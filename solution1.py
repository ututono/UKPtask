import torch
from torch import nn
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import os
import sys
from multiprocessing import cpu_count
import torch.nn.functional as F
import matplotlib.pyplot as plt


# %config InlineBackend.figure_format = 'retina'
# plt.style.use('seaborn')

def run_CPU():
    cpu_num = cpu_count()  # obtain the maximum number of cpu kernels
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)


class TensorDataset(Dataset):
    def __init__(self, data_tensor, labels_tensor):
        self.data_tensor = data_tensor
        self.labels_tensor = labels_tensor

    def __getitem__(self, item):
        return self.data_tensor[item], self.labels_tensor[item]

    def __len__(self):
        return self.data_tensor.size(0)


class Data_Prepare:
    def __init__(self):
        # public variables
        self.EMBED_DIM = 50
        self.vocab = self.load_glove_model('word_embedding/glove.6B.50d/glove.6B.50d.txt')

    def load_glove_model(self, File):
        print("Loading Glove Model")
        glove_model = {}
        with open(File, 'r', encoding='utf-8') as f:
            for line in f:
                split_line = line.split()
                word = split_line[0]
                embedding = np.array(split_line[1:], dtype=np.float64)
                glove_model[word] = embedding
        print(f"{len(glove_model)} words loaded!")
        return glove_model

    def extract_word_labels(self, filepath):
        df = pd.read_csv(filepath, delimiter='\t', names=['Word', 'POS', 'NP', 'NER'], skiprows=[0])
        df = self.word_embedding(df)
        labels = self.onehot_encode(df.NER)
        word_embeddings = np.array(df.Embedding.to_list())
        labels_tensor = torch.Tensor(np.array(labels))
        data_tensor = torch.Tensor(word_embeddings)
        dataset = TensorDataset(data_tensor, labels_tensor)

        return dataset

    def word_embedding(self, df):
        embeddings = []
        for word in df.Word:
            vector = self.vocab.get(str(word).lower())
            if vector is not None:
                embeddings.append(vector)
            else:
                embeddings.append(np.zeros(self.EMBED_DIM))

        df['Embedding'] = embeddings
        return df

    def onehot_encode(self, labels):
        labels_to_ids = {k: v for v, k in enumerate(set(labels.to_list()))}
        result = []
        for label in labels:
            vec = np.zeros(len(labels_to_ids))
            vec[labels_to_ids.get(label)] = 1
            result.append(vec)
        return result


class RNN(nn.Module):
    def __init__(self, params: dict):
        super(RNN, self).__init__()
        self.num_layers = params['num_layers']
        self.hidden_size = params['hidden_size']
        self.batch_size = params['batch_size']
        self.bilstm = nn.LSTM(input_size=params['input_size'],
                              hidden_size=params['hidden_size'] // 2,
                              num_layers=params['num_layers'],
                              batch_first=True,
                              bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.classifier = nn.Linear(params['hidden_size'], params['num_classes'])

        self.hidden = self.init_hidden()

    def forward(self, x):
        # self.hidden = self.init_hidden()
        out, _ = self.bilstm(x)
        out = out.view(-1, self.hidden_size)
        out = self.classifier(out)
        return out

    def init_hidden(self):
        return (torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_size // 2),
                torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_size // 2))


def train(model, device, criterion, train_loader, dev_loader, optimizer, epoch):
    def process_bar(num, total):
        rate = float(num) / total
        ratenum = int(100 * rate)
        r = '\rModel training:[{}{}]{}%'.format('*' * ratenum, ' ' * (100 - ratenum), ratenum)
        sys.stdout.write(r)
        sys.stdout.flush()

    min_valid_loss = np.inf

    model.train()  # set the model to train mode
    log_interval = int(len(train_loader) * 0.01)
    train_loss=0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()   # Clear the gradients
        output = model(data)  # Forward Pass
        loss = criterion(output, target)  #  Find the Loss
        loss.backward()
        optimizer.step()  # update
        train_loss +=loss.item()
        if batch_idx % log_interval == 0:  # output the log of training based on predefined interval
            process_bar(batch_idx, len(train_loader))

    valid_loss = 0.
    rnn.eval()
    targets = []
    golds = []
    for data, label in dev_loader:
        if torch.cuda.is_available():
            data, label = data.cuda(), label.cuda()

        target = rnn(data)
        loss = criterion(target, label)
        valid_loss = loss.item() * data.size(0)

        targets.append(torch.argmax(target).item())
        golds.append(torch.argmax(label).item())

    f1_scores_macro = f1_score(golds, targets, average='macro')
    print("\nMacro average F1_score is{0}".format(f1_scores_macro))

    f1_score_micro = f1_score(golds, targets, average='micro')
    print("Micro average F1_score is{0}".format(f1_score_micro))

    print("\nEpochs: {0} / {1}: Training Loss:{2} \t\t Validation Loss:{3}"
          .format(epoch, params['epochs'], train_loss / len(train_dataloader), valid_loss / len(dev_dataset)))
    if min_valid_loss > valid_loss:
        print('Validation Loss Decreased({0:.6f}--->{1:.6f}) \t Saving The Model'.format(min_valid_loss, valid_loss))
        min_valid_loss = valid_loss
        # Saving State Dict
        torch.save(rnn.state_dict(), 'saved_model.pth')

        # if batch_idx % log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, criterion):
    model.eval()  # set model to evaluation mode
    test_loss = 0
    correct = 0

    with torch.no_grad():  # stop gradient update
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            # print(output.max(1, keepdim=True)[1])
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.max(1, keepdim=True)[1]).sum().item()  # count the correct predictions

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    run_CPU()

    tran_path = 'data/train.conll'
    dev_path = 'data/dev.conll'
    test_path = 'data/test.conll'

    data_prepare = Data_Prepare()
    train_dataset = data_prepare.extract_word_labels(tran_path)
    dev_dataset = data_prepare.extract_word_labels(dev_path)
    test_dataset = data_prepare.extract_word_labels(test_path)

    params = {
        'num_layers': 1,
        'hidden_size': 100,
        'input_size': 50,
        'learning_rate': 0.01,
        'optimizer': 'adam',
        'epochs': 10,
        'batch_size': 1,
        'num_classes': 10,
        'dropout': 0.1
    }
    torch.manual_seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rnn = RNN(params=params)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(rnn.parameters(), lr=params['learning_rate'])

    # train_mini_set = train_dataset
    train_mini_set = train_dataset[:10000]
    train_mini_set = TensorDataset(train_mini_set[0], train_mini_set[1])

    train_dataloader = DataLoader(train_mini_set,
                                  batch_size=params['batch_size'],
                                  shuffle=False,
                                  num_workers=0)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=params['batch_size'],
                                 shuffle=False,
                                 num_workers=0)

    dev_dataloader = DataLoader(dev_dataset,
                                 batch_size=params['batch_size'],
                                 shuffle=False,
                                 num_workers=0)
    print("-"*30+'The architecture of the rnn'+"-"*30)
    print(rnn)
    print("-" * 80)

    for epoch in range(1, params['epochs'] + 1):
        train(rnn, device,
              criterion=criterion,
              train_loader=train_dataloader,
              dev_loader=dev_dataloader,
              optimizer=optimizer,
              epoch=epoch)

    print('Training over.')

    test(rnn, device, test_dataloader, criterion)
