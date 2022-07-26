import torch 
import numpy as np 
import gensim

word2vec_path = "wiki.vi.model.bin"
word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
padding_token = np.resize(word2vec['pad'], (1,400))
unk_token = np.resize(word2vec['unk'], (1,400))


def make_word2vec_vector_cnn(args, sentence):
    padded_X = [padding_token for i in range(args.max_sen_len)]
    i = 0
    for word in sentence:
        if word not in word2vec:
            padded_X[i] = args.unk_token
            print(word)
        else:
            padded_X[i] = np.resize(word2vec['unk'], (1,400))
    out = torch.Tensor(np.concatenate(padded_X))
    out.required_grad = False
    return out


def create_batch_data(args, X, Y):
    batches_X = []
    batches_Y = []
    batches = []
    batch_size = args.batch_size
    
    batch_X = []
    batch_Y = []
    for idx, _ in enumerate(X,1):
        if idx % batch_size == 0 and idx > 0:
            batch_X.append(X[idx])
            batch_Y.append(Y[idx])
            batches_X.append(batch_X)
            batches_Y.append(batch_Y)
            batches.append([batch_X, batch_Y])
            batch_X = []
            batch_Y = []
        else:
            if idx < len(X):
                batch_X.append(X[idx])
                batch_Y.append(Y[idx])
    return batches

            
def make_word2vec_vector_cnn(args, batch):
    batch_token = []
    batch_label = []

    batch_label = torch.Tensor(np.array(batch[1]))
    batch_label = batch_label.to('cuda:0')
    for sentence in batch[0]:
        padded_X = [padding_token for i in range(args.max_sen_len)]
        i = 0
        for idx, word in enumerate(sentence):
            if idx < args.max_sen_len:
                if word not in word2vec:
                    padded_X[i] = unk_token
                else:
                    padded_X[i] = np.resize(word2vec['unk'], (1,400))
        out = np.concatenate(padded_X)

        batch_token.append(out)
    batch_token = np.stack(batch_token, axis = 0)
    batch_token = torch.Tensor(batch_token)
    batch_token = batch_token.to('cuda:0')
    batch_token.required_grad = False
    batch[0] = batch_token
    batch[1] = batch_label
    return batch
