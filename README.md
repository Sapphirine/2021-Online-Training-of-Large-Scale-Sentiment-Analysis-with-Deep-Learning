# Online Training of Large-Scale Sentiment Analysis with Deep Learning

This repository is the code of project Online Training of Large-Scale Sentiment Analysis with Deep Learning.

## Requirements

Python3 should be installed. Besides, the following packages are in need.

- Pytorch 
- sklearn

## Usage

First unzip the `imdb.zip` into `imdb` directory to get the data.
It is zipped because the data file is larger than 100MB.

Run `rnn.py` as following to reproduce the results in the report.

```
python rnn.py [--alpha alpha] [trainer]
```

`trainer` should be one of `baseline`, `decay`, `back`, and `drop`, corresponding to the four learning strategies in the project.

`alpha` could be given if the trainer is `drop`.

After training process, a graph would be printed to show the results.
