from copyreg import pickle
import torch 
from config import Param
from model import CNN_NLP
import numpy as np
from prepare_data import Process_data
import pandas as pd
from utils import create_batch_data
import torch.nn as nn
import torch.optim as optim
from train_model import train
def run(args):
    import gensim

    word2vec_path = "wiki.vi.model.bin"
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    padding_token = np.resize(word2vec['pad'], (1,400))
    unk_token = np.resize(word2vec['unk'], (1,400))
    lbs = ['Chinh tri Xa hoi',  'Khoa hoc', 'Phap luat',  'The gioi',  'Van hoa', 'Doi song', 'Kinh doanh', 'Suc khoe',   'The thao',  'Vi tinh']

    if args.is_saved: 
        with open('x_train.pkl', 'rb') as f:
            X_train = pickle.load(f)
        with open('x_test.pkl', 'rb') as f:
            X_test = pickle.load(f)
        with open('y_train.pkl', 'rb') as f:
            Y_train = pickle.load(f)
        with open('y_test.pkl', 'rb') as f:
            Y_test = pickle.load(f)
    else:
    # prepare tran, test data
        process_train = Process_data()
        process_test = Process_data(is_train = False)

        train_df = process_train.df
        test_df = process_test.df

        X_train, X_test = list(train_df['text'].values), list(test_df['text'].values)
        Y_train, Y_test = list(train_df['labels2id'].values), list(test_df['labels2id'].values)

    if args.save_processed_data:
        with open('x_train.pkl', 'wb') as f:
            pickle.dump(X_train, f)
        with open('x_test.pkl', 'wb') as f:
            pickle.dump(X_test, f)
        with open('y_train.pkl', 'wb') as f:
            pickle.dump(Y_train, f)
        with open('y_test.pkl', 'wb') as f:
            pickle.dump(Y_test, f)

    train_dataloader = create_batch_data(X_train, Y_train, batch_size = args.batch_size)
    val_dataloader = create_batch_data(X_test, Y_test, batch_size = args.batch_size)

    NUM_CLASSES = len(lbs)

    cnn_model = CNN_NLP()
    cnn_model.to('cuda:0')
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=args.lr)
    train(args, cnn_model, optimizer, train_dataloader, val_dataloader, word2vec, padding_token, unk_token)


if __name__ == '__main__':
    param = Param()
    args = param.args
    run(args=args)