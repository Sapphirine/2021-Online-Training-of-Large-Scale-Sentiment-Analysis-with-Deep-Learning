import pandas as pd
import math

def read_data_file(filename):
	return pd.read_csv(filename, names=['user', 'item', 'label', 'text'], sep='\t\t')

def read_split_data(dataset):
	return (read_data_file('%s/%s.ss' % (dataset, split)) for split in ['train', 'test', 'dev'])

def read_combined_data(dataset):
	return pd.concat(read_split_data(dataset)).sample(frac=1, random_state=0)

class BaselineGenerator:
	def __init__(self, data, batch_size):
		self.data = data
		self.batch_size = batch_size
		self.i = 0

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

	def feedback(self, logit):
		return


if __name__ == '__main__':
	data = read_combined_data('imdb')
	from collections import Counter
	print(Counter(data['label']))