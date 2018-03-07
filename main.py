from argparse import ArgumentParser

import torch
from torch import optim

from torchtext import data
from torchtext import datasets

from model import LSTMJump


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--R', type=int, default=20)
    parser.add_argument('--K', type=int, default=40)
    parser.add_argument('--N', type=int, default=5)
    return parser.parse_args()


def main(args):
    TEXT = data.Field(lower=True)
    LABEL = data.Field(sequential=False)

    train, test = datasets.IMDB.splits(TEXT, LABEL)

    TEXT.build_vocab(train)
    LABEL.build_vocab(train)

    train_iter, test_iter = data.BucketIterator.splits(
        (train, test), batch_sizes=(50, 50), device=args.gpu, repeat=False)

    model = LSTMJump(len(TEXT.vocab), 32, 128, len(LABEL.vocab), args.N, args.K, args.R)
    model.cuda(args.gpu)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    max_accuracy = 0
    for i in range(1000):
        print('Epoch: {}'.format(i + 1))
        sum_loss = 0
        model.train()
        for batch in train_iter:
            optimizer.zero_grad()
            loss = model(batch.text, batch.label)
            sum_loss += loss.data[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 1)
            optimizer.step()
        print('Loss: {}'.format(sum_loss / len(train_iter)))
        sum_correct = 0
        total = 0
        model.eval()
        for batch in test_iter:
            y = model.inference(batch.text)
            sum_correct += y.eq(batch.label).sum().float()
            total += batch.label.size(0)
        accuracy = (sum_correct / total).data[0]
        max_accuracy = max(accuracy, max_accuracy)
        print('Accuracy: {}'.format(accuracy))
    print('Max Accuracy: {}'.format(max_accuracy))


if __name__ == '__main__':
    args = parse_args()
    main(args)
