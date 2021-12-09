from tqdm import tqdm
import math
from itertools import chain
import torch
from sklearn.metrics import accuracy_score, f1_score
from matplotlib import pyplot as plt
import pandas as pd

LEARNING_RATE = 1e-3

class Trainer:
    def __init__(self):
        self.predicts = []
        self.labels = []
        self.accs = []
        self.recent_accs = []
        self.f1s = []
        self.recent_f1s = []
        self.n_recent = 1000
        self.figure = plt.figure()

    def step(self):
        acc = accuracy_score(self.predicts, self.labels)
        recent_acc = accuracy_score(self.predicts[-self.n_recent:], self.labels[-self.n_recent:])
        self.recent_accs.append(recent_acc)
        self.accs.append(acc)
        f1 = f1_score(self.predicts, self.labels, average='macro')
        recent_f1 = f1_score(self.predicts[-self.n_recent:], self.labels[-self.n_recent:], average='macro')
        self.recent_f1s.append(recent_f1)
        self.f1s.append(f1)

    def plot(self, title):
        self.figure.clf()
        self.figure.suptitle(title)
        ax_acc, ax_f1 = self.figure.subplots(1, 2)

        ax_acc.plot(self.accs, label='overall accuracy', c='r')
        ax_acc.plot(self.recent_accs, label='recent accuracy', c='b')
        ax_acc.set_xlabel('iteration')
        ax_acc.set_ylabel('accuracy')
        ax_acc.legend()

        ax_f1.plot(self.f1s, label='overall f1-score', c='r')
        ax_f1.plot(self.recent_f1s, label='recent f1-score', c='b')
        ax_f1.set_xlabel('iteration')
        ax_f1.set_ylabel('f1-score')
        ax_f1.legend()

        plt.pause(.05)

    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)

    def __iter__(self):
        return self

    def __next__(self):
        cur_data = self.data.iloc[self.i : self.i + self.batch_size].copy()
        if len(cur_data) == 0:
            raise StopIteration
        self.i += self.batch_size
        cur_data['train-text'] = cur_data['text']
        cur_data['train-label'] = cur_data['label']
        return cur_data


class BaselineTrainer(Trainer):
    """
    Acc is 0.27187084162554903
    Acc in the last 1000 steps is 0.36
    Acc in the last 2000 steps is 0.353
    Acc in the last 3000 steps is 0.349
    f1 is 0.15285019141667655
    f1 in the last 1000 steps is 0.22008621996269517
    f1 in the last 2000 steps is 0.22940647545229167
    f1 in the last 3000 steps is 0.23468542396577372
    """
    def __init__(self, model, data, batch_size, realtime_figure):
        super(BaselineTrainer, self).__init__()
        self.model = model
        self.data = data
        self.batch_size = batch_size
        self.i = 0
        self.realtime_figure = realtime_figure

    def train(self):
        self.optimizer = torch.optim.Adam(chain(self.model.parameters()), lr=LEARNING_RATE)
        for i, batch_data in enumerate(tqdm(self)):
            # online predict
            logit, _ = self.model.predict_batch(batch_data)
            self.predicts += logit.argmax(-1).tolist()
            self.labels += batch_data['label'].tolist()

            # train
            _, loss = self.model.train_batch(batch_data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.step()
            if self.realtime_figure and i > 0 and i % 100 == 0:
                self.plot('naive')
            if i % 100 == 0:
                print('Step', i, 'overall acc =', self.accs[-1], ', recent acc =', self.recent_accs[-1], ', overall f1 =', self.f1s[-1], ', recent f1 =', self.recent_f1s[-1])

        self.plot('naive')
        plt.show()
        return self.predicts, self.labels

class DecayTrainer(Trainer):
    """
    Acc is 0.2815742060080783
    Acc in the last 1000 steps is 0.372
    Acc in the last 2000 steps is 0.366
    Acc in the last 3000 steps is 0.35433333333333333
    f1 is 0.1685072298700095
    f1 in the last 1000 steps is 0.2752111501419746
    f1 in the last 2000 steps is 0.2695350718147799
    f1 in the last 3000 steps is 0.26615958029445086
    """
    def __init__(self, model, data, batch_size, realtime_figure):
        super(DecayTrainer, self).__init__()
        self.model = model
        self.data = data
        self.batch_size = batch_size
        self.i = 0
        self.realtime_figure = realtime_figure

    def train(self):
        self.optimizer = torch.optim.Adam(
            chain(self.model.parameters()), 
            lr=LEARNING_RATE,
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=.99999)
        for i, batch_data in enumerate(tqdm(self)):
            # online predict
            logit, _ = self.model.predict_batch(batch_data)
            self.predicts += logit.argmax(-1).tolist()
            self.labels += batch_data['label'].tolist()

            # train
            _, loss = self.model.train_batch(batch_data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            self.step()
            if self.realtime_figure and i > 0 and i % 100 == 0:
                self.plot('decaying learning rate')
            if i % 100 == 0:
                print('Step', i, 'overall acc =', self.accs[-1], ', recent acc =', self.recent_accs[-1], ', overall f1 =', self.f1s[-1], ', recent f1 =', self.recent_f1s[-1])

        self.plot('decaying learning rate')
        plt.show()
        return self.predicts, self.labels

class BackTrainer(Trainer):
    """
    Acc is 0.3010398144113803
    Acc in the last 1000 steps is 0.385
    Acc in the last 2000 steps is 0.381
    Acc in the last 3000 steps is 0.37366666666666665
    f1 is 0.19340976277361452
    f1 in the last 1000 steps is 0.29892470833958384
    f1 in the last 2000 steps is 0.2911842800481547
    f1 in the last 3000 steps is 0.28108738421136986
    """
    def __init__(self, model, data, batch_size, realtime_figure):
        super(BackTrainer, self).__init__()
        self.model = model
        self.data = data
        self.batch_size = batch_size
        self.i = 0
        self.realtime_figure = realtime_figure
        self.seen = pd.DataFrame()

    def train(self):
        self.optimizer = torch.optim.Adam(
            chain(self.model.parameters()), 
            lr=LEARNING_RATE,
        )
        for i, batch_data in enumerate(tqdm(self)):
            self.seen = self.seen.append(batch_data)
            # online predict
            logit, _ = self.model.predict_batch(batch_data)
            self.predicts += logit.argmax(-1).tolist()
            self.labels += batch_data['label'].tolist()

            # train
            batch_data = batch_data.append(self.seen.sample(self.batch_size))
            _, loss = self.model.train_batch(batch_data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.step()
            if self.realtime_figure and i > 0 and i % 100 == 0:
                self.plot('looking backward')
            if i % 100 == 0:
                print('Step', i, 'overall acc =', self.accs[-1], ', recent acc =', self.recent_accs[-1], ', overall f1 =', self.f1s[-1], ', recent f1 =', self.recent_f1s[-1])

        self.plot('looking backward')
        plt.show()
        return self.predicts, self.labels

class DropTrainer(Trainer):
    """
    Acc is 0.30414865931063717
    Acc in the last 1000 steps is 0.404
    Acc in the last 2000 steps is 0.3975
    Acc in the last 3000 steps is 0.39166666666666666
    f1 is 0.2071293151033992
    f1 in the last 1000 steps is 0.32789087819307594
    f1 in the last 2000 steps is 0.31621607156857573
    f1 in the last 3000 steps is 0.32170351901726935
    """
    def __init__(self, model, data, batch_size, realtime_figure, alpha):
        super(DropTrainer, self).__init__()
        self.model = model
        self.data = data
        self.batch_size = batch_size
        self.i = 0
        self.realtime_figure = realtime_figure
        self.seen = pd.DataFrame()
        self.alpha = alpha

    def train(self):
        self.optimizer = torch.optim.Adam(
            chain(self.model.parameters()), 
            lr=LEARNING_RATE,
        )
        for i, batch_data in enumerate(tqdm(self)):
            self.seen = self.seen.append(batch_data)
            # online predict
            logit, _ = self.model.predict_batch(batch_data)
            self.predicts += logit.argmax(-1).tolist()
            self.labels += batch_data['label'].tolist()

            batch_data = batch_data.append(self.seen.sample(self.batch_size))
            logit, _ = self.model.predict_batch(batch_data)
            cur_prob = torch.gather(torch.softmax(logit, dim=-1), -1, torch.LongTensor(batch_data['label'].tolist()).reshape(-1, 1))

            index = (cur_prob < self.alpha).reshape(-1).tolist()
            batch_data = batch_data.iloc[index]
            dropped = sum(map(lambda x: 1 - x, index))
            if dropped > 0:
                print('dropped', dropped, 'samples')
            else:
                print(cur_prob.max())

            # train
            _, loss = self.model.train_batch(batch_data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.step()
            if self.realtime_figure and i > 0 and i % 100 == 0:
                self.plot('dropping easy samples')
            if i % 100 == 0:
                print('Step', i, 'overall acc =', self.accs[-1], ', recent acc =', self.recent_accs[-1], ', overall f1 =', self.f1s[-1], ', recent f1 =', self.recent_f1s[-1])

        self.plot('dropping easy samples')
        plt.show()
        return self.predicts, self.labels
