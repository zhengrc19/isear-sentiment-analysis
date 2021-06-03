# ISEAR Dataset Sentiment Analysis

This is a Python script to preprocess, build, train and test the ISEAR dataset on sentiment analysis. It has implemented:

- a CNN model
- an LSTM model
- an MLP model used as the baseline model. 

The architecture of the CNN model is the one described in the paper [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) (Kim, 2014), although the convolutional kernel sizes and the number of features could be altered. 

The architecture of the LSTM model consists of an embedding layer, a (optionally) bidirectional LSTM layer and a fully connected layer. The number of stacked LSTM cells, the number of features in the hidden state, and whether or not to use bidirectional LSTM can be specified in command line arguments.

The MLP model consists of two hidden fully connected layers and is meant to be used as a baseline model.



## Run train script

A simple command of 

```bash
python train.py
```

will run the training script on the CNN model.  

To specify which model to use, or other parameters, refer to the following table.

| Parameter Name         | Type    | Default value  | Description                                                  |
| ---------------------- | ------- | -------------- | ------------------------------------------------------------ |
| `--model`              | `str`   | `'cnn'`        | Which model to train. Currently, can choose between "cnn", "lstm", and "mlp". |
| `--epoch`              | `int`   | `60`           | Number of epochs in training.                                |
| `--batch_size`         | `int`   | `64`           | Batch size.                                                  |
| `--data_dir`           | `str`   | `'./data'`     | Directory for train, validation, and test set csv files.     |
| `--emb_dim`            | `int`   | `300`          | Word embedding dimension.                                    |
| `--lr`                 | `float` | `0.5`          | Learning rate.                                               |
| `--weight_decay`       | `float` | `0.01`         | Weight decay (L2 regularization) constant.                   |
| `--gamma`              | `float` | `0.1`          | Multiplicative factor of learning rate decay.                |
| `--step_size`          | `float` | `15`           | Period of learning rate decay.                               |
| `--dropout`            | `float` | `0.5`          | During training, randomly zeroes some of the elements of the input tensor with probability p using samples from a Bernoulli distribution. |
| `--lstm_hidden_dim`    | `int`   | `64`           | Number of features in LSTM hidden state.                     |
| `--lstm_num_layers`    | `int`   | `1`            | Number of recurrent layers of LSTM. E.g., setting `num_layers=2` would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results. |
| `--lstm_bidirectional` | `bool`  | `True`         | If `True`, becomes a bidirectional LSTM.                     |
| `--cnn_num_features`   | `int`   | `200`          | Number of channels (features) produced by the convolution.   |
| `--cnn_window_sizes`   | `list`  | `[2, 3, 4, 5]` | Size of the convolving kernel, or in 1d terms window size. To use multiple window sizes, run like so: `python train.py --model cnn --cnn_window_sizes 2 3 4 5` |



## Requirements

The following packages are required, and are given in `requirements.txt`:

```text
torch==1.8.1
torchtext==0.9.1
tqdm
pandas
scikit-learn
matplotlib
```

Using the command

```bash
pip install -r requirements.txt
```

will install all these packages automatically.

