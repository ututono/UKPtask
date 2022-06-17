import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import sys
from ConfusionMatrix import ConfusionMatrix  # customized class used to evaluate the model

class Trainer:
    def __init__(self,optimizer:torch.optim.Optimizer):
        self.optimizer=optimizer
        pass

    def process_bar(self,num, total):
        rate = float(num) / total
        ratenum = int(100 * rate)
        r = '\rModel training:[{}{}]{}%'.format('*' * ratenum, ' ' * (100 - ratenum), ratenum)
        sys.stdout.write(r)
        sys.stdout.flush()

    def run(self,model, train, loss, batch_size,labels,optimizer=None):
        model.train()
        self.optimizer= optimizer if optimizer else self.optimizer
        cm=ConfusionMatrix(labels)

        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
        log_interval = int(len(train_loader) * 0.01)
        for batch_idx, batch in enumerate(train_loader):

            loss_value, y_pred, y_actual = self.update(model, loss, batch,self.optimizer)
            yt = y_actual.cpu().int().numpy()
            yp = y_pred.cpu().int().numpy()
            cm.add_batch(yt,yp)
            if batch_idx % log_interval == 0:
                self.process_bar(batch_idx, len(train_loader))

        print(cm.get_all_metrics())
        return cm

    def update(self,model, loss, batch, optimizer):
            optimizer.zero_grad()
            x, lengths, y = batch
            # print(batch)
            lengths, perm_idx = lengths.sort(0, descending=True)
            x_sorted = x[perm_idx]
            y_sorted = y[perm_idx]
            # print(x_sorted,lengths,y_sorted)
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            y_sorted = y_sorted.to(device)
            inputs = (x_sorted.to(device), lengths)
            # print(inputs)
            y_pred = model(inputs)

            mask = (y_sorted != 0)
            valid = (mask.sum(dim=1))
            y_sorted=y_sorted[mask].split(valid.tolist())[0]

            y_sorted=np.array(y_sorted).tolist()
            y_sorted=torch.tensor(y_sorted, dtype=torch.long)
            y_pred=(torch.tensor(y_pred[1], dtype=torch.long))
            loss_value=model.neg_log_likelihood(inputs, y_sorted)
            # loss_value = loss(torch.tensor(y_pred[1],dtype= torch.long), torch.tensor(y_sorted,dtype= torch.long))
            loss_value.backward()
            optimizer.step()
            return loss_value.item(), y_pred, y_sorted

class Evaluator:
    def __init__(self):
        pass

    def run(self,model, dataset, labels, batch_size=1):
        model.train()

        cm=ConfusionMatrix(labels)

        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        log_interval = int(len(train_loader) * 0.01)
        for batch_idx, batch in enumerate(train_loader):

            y_pred, y_actual = self.inference(model, batch)
            yt = y_actual.cpu().int().numpy()
            yp = y_pred.cpu().int().numpy()
            cm.add_batch(yt,yp)
            # if batch_idx % log_interval == 0:
            #     process_bar(batch_idx, len(train_loader))

        print(cm.get_all_metrics())
        return cm

    def inference(self, model, batch):
        with torch.no_grad():
            x, lengths, y = batch
            lengths, perm_idx = lengths.sort(0, descending=True)
            x_sorted = x[perm_idx]
            y_sorted = y[perm_idx]
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            y_sorted = y_sorted.to(device)
            inputs = (x_sorted.to(device), lengths)
            y_pred = model(inputs)

            mask = (y_sorted != 0)
            valid = (mask.sum(dim=1))
            y_sorted=y_sorted[mask].split(valid.tolist())[0]

            y_sorted=np.array(y_sorted).tolist()
            y_sorted=torch.tensor(y_sorted, dtype=torch.long)
            y_pred=(torch.tensor(y_pred[1], dtype=torch.long))

            return y_pred, y_sorted
