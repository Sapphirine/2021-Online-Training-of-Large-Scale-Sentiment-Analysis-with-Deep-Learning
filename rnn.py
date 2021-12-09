from data import *
import torch
from sklearn.metrics import accuracy_score, f1_score
from trainer import *
from sys import argv

if len(argv) < 2:
	print("""Usage: python rnn.py [--realtime-figure] [--alpha alpha] [trainer]""")
	exit(0)

TRAINER = argv[-1]
DATASET = 'imdb'
BATCH_SIZE = 64
REALTIME_FIGURE = '--realtime-figure' in argv
ALPHA = float(argv[argv.index('--alpha') + 1]) if 'alpha' in argv else .7

data = read_combined_data(DATASET)

word2num = {'[PAD]': 0}


def tokenize(text):
    for word in text.split():
        if word not in word2num:
            word2num[word] = len(word2num)
    return text.split()


data['text'] = data['text'].apply(tokenize)
data['text'] = data['text'].apply(lambda text: [word2num[word] for word in text])
data['label'] -= 1


class RNNModel(torch.nn.Module):

    def __init__(self):
        super(RNNModel, self).__init__()

        self.embed = torch.nn.Embedding(len(word2num), 128)
        self.rnn = torch.nn.RNN(input_size=128, hidden_size=128)
        self.linear = torch.nn.Linear(128, 10)
        self.loss_function = torch.nn.NLLLoss()

    def forward(self, x):
        x = self.embed(x).transpose(0, 1)
        x = self.rnn(x)[0].transpose(0, 1).transpose(1, 2)
        x = torch.nn.functional.max_pool1d(x, x.size(2)).squeeze()
        logit = self.linear(x)
        return torch.nn.functional.log_softmax(logit, dim=-1)

    def train_batch(self, batch_data):
        batch_x = batch_data['train-text'].tolist()
        max_len = max(len(x) for x in batch_x)
        batch_x = torch.LongTensor([x + [0] * (max_len - len(x)) for x in batch_x])
        batch_y = torch.LongTensor(batch_data['train-label'].tolist())
        
        logit = self(batch_x)
        loss = self.loss_function(logit, batch_y)
        return logit, loss

    def predict_batch(self, batch_data):
        # online predict
        batch_x = batch_data['text'].tolist()
        max_len = max(len(x) for x in batch_x)
        batch_x = torch.LongTensor([x + [0] * (max_len - len(x)) for x in batch_x])
        batch_y = torch.LongTensor(batch_data['label'].tolist())

        logit = self(batch_x)
        loss = self.loss_function(logit, batch_y)
        return logit, loss


model = RNNModel()

if TRAINER == 'baseline':
	trainer = BaselineTrainer(model, data, BATCH_SIZE, REALTIME_FIGURE)
elif TRAINER == 'decay':
	trainer = DecayTrainer(model, data, BATCH_SIZE, REALTIME_FIGURE)
elif TRAINER == 'back':
	trainer = BackTrainer(model, data, BATCH_SIZE, REALTIME_FIGURE)
elif TRAINER == 'drop':
	trainer = DropTrainer(model, data, BATCH_SIZE, REALTIME_FIGURE, ALPHA)
else:
	print('Invalid trainer:', TRAINER)
	exit(1)
predicts, labels = trainer.train()

acc = accuracy_score(predicts, labels)
acc_last1000 = accuracy_score(predicts[-1000:], labels[-1000:])
acc_last2000 = accuracy_score(predicts[-2000:], labels[-2000:])
acc_last3000 = accuracy_score(predicts[-3000:], labels[-3000:])

f1 = f1_score(predicts, labels, average='macro')
f1_last1000 = f1_score(predicts[-1000:], labels[-1000:], average='macro')
f1_last2000 = f1_score(predicts[-2000:], labels[-2000:], average='macro')
f1_last3000 = f1_score(predicts[-3000:], labels[-3000:], average='macro')

print('Acc is', acc)
print('Acc in the last 1000 steps is', acc_last1000)
print('Acc in the last 2000 steps is', acc_last2000)
print('Acc in the last 3000 steps is', acc_last3000)

print('f1 is', f1)
print('f1 in the last 1000 steps is', f1_last1000)
print('f1 in the last 2000 steps is', f1_last2000)
print('f1 in the last 3000 steps is', f1_last3000)
