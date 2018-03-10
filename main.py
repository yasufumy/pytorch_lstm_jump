from argparse import ArgumentParser

import torch
from torch import optim

from torchtext import data
from torchtext import datasets
from utils import get_word2vec

from model import LSTMJump


PAD_TOKEN = '<pad>'


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--batch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--R', type=int, default=20)
    parser.add_argument('--K', type=int, default=40)
    parser.add_argument('--N', type=int, default=5)
    return parser.parse_args()


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


def main(args):
    TEXT = data.Field(lower=True, postprocessing=pick_fix_length(400), pad_token=PAD_TOKEN)
    LABEL = data.Field(sequential=False, pad_token=None, unk_token=None)

    train, test = datasets.IMDB.splits(TEXT, LABEL)

    TEXT.build_vocab(train)
    LABEL.build_vocab(train)

    train_iter, test_iter = data.BucketIterator.splits(
        (train, test), batch_sizes=(args.batch, args.batch * 4), device=args.gpu, repeat=False)

    model = LSTMJump(len(TEXT.vocab), 300, 128, len(LABEL.vocab),
                     args.R, args.K, args.N, 80, 8)
    model.load_pretrained_embedding(
        get_word2vec(TEXT.vocab.itos, '.vector_cache/GoogleNews-vectors-negative300.bin'))
    model.cuda(args.gpu)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    max_accuracy = 0
    for i in range(1000):
        print('Epoch: {}'.format(i + 1))
        sum_loss = 0
        model.train()
        for batch in train_iter:
            optimizer.zero_grad()
            loss = model(batch.text, batch.label)
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 1.)
            optimizer.step()
            sum_loss += loss.data[0]
        print(f'Loss: {sum_loss / len(train_iter)}')
        sum_correct = 0
        total = 0
        model.eval()
        for batch in test_iter:
            y = model.inference(batch.text)
            sum_correct += y.eq(batch.label).sum().float()
            total += batch.label.size(0)
        accuracy = (sum_correct / total).data[0]
        max_accuracy = max(accuracy, max_accuracy)
        print(f'Accuracy: {accuracy}')
    print(f'Max Accuracy: {max_accuracy}')


if __name__ == '__main__':
    args = parse_args()
    main(args)
