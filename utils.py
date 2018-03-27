import os.path as osp

import gensim
import torch


def get_word2vec(itos, path='GoogleNews-vectors-negative300.bin', cache=True):
    path_pt = path + '.pt'
    if not osp.isfile(path_pt):
        model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
        dim = 300
        vectors = torch.Tensor(len(itos), dim)
        for i, token in enumerate(itos):
            if token in model:
                vectors[i] = torch.from_numpy(model[token]).view(-1, dim)
            else:
                vectors[i] = torch.Tensor(1, dim).zero_()
        if cache:
            torch.save(vectors, path_pt)
    else:
        vectors = torch.load(path_pt)
    return vectors


def pick_fix_length(length):
    import random

    def _pick(arr, vocab, train):
        if train:
            max_length = len(arr[0])
            pad_id = vocab.stoi[PAD_TOKEN]
            if max_length > length:
                result = []
                for ex in arr:
                    real_length = len([x for x in ex if x != pad_id])
                    if real_length > length:
                        n = random.randrange(real_length - length + 1)
                        result.append(ex[n:n + length])
                    else:
                        result.append(ex[:length])
                return result
            else:
                return [ex + [pad_id] * (length - max_length) for ex in arr]
        else:
            return arr
    return _pick
