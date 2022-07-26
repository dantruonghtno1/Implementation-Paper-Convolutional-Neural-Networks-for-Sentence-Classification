import torch 
import numpy as np 
import gensim

word2vec_path = "wiki.vi.model.bin"
word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
padding_token = np.resize(word2vec['pad'], (1,400))
unk_token = np.resize(word2vec['unk'], (1,400))


def make_word2vec_vector_cnn(args, sentence):
    padded_X = [args.padding_token for i in range(args.max_sen_len)]
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