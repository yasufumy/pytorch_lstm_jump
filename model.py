from operator import setitem

import torch
from torch import nn
from torch.nn import functional
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.nn.utils.rnn import pack_padded_sequence


class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, categories):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size)
        self.output = nn.Linear(hidden_size, categories)
        self.nll_loss = nn.NLLLoss()
        self.hidden_size = hidden_size

    def load_pretrained_embedding(self, embedding):
        self.embed.weight = nn.Parameter(embedding)

    def forward(self, xs, lengths, t):
        _, batch_size = xs.size()
        embs = functional.dropout(self.embed(xs), 0.1)
        lengths = lengths.view(-1).tolist()
        packed_embs = pack_padded_sequence(embs, lengths)
        hs, (h, c) = self.lstm(packed_embs)
        y = functional.log_softmax(self.output(h.view(batch_size, -1)), dim=1)
        return self.nll_loss(y, t)

    def inference(self, xs, lengths):
        _, batch_size = xs.size()
        embs = self.embed(xs)
        lengths = lengths.view(-1).tolist()
        packed_embs = pack_padded_sequence(embs, lengths)
        hs, (h, c) = self.lstm(packed_embs)
        return self.output(h.view(batch_size, -1)).max(dim=1)[1]


class LSTMJump(LSTM):
    def __init__(self, vocab_size, embed_size, hidden_size, categories,
                 R=20, K=40, N=5, R_test=80, N_test=8):
        super().__init__(vocab_size, embed_size, hidden_size, categories)
        self.linear = nn.Linear(hidden_size, K + 1)
        self.baseline = nn.Linear(hidden_size, 1)
        self.mse_loss = nn.MSELoss(size_average=False)
        self._R_train = R
        self._R_test = R_test
        self._N_train = N
        self._N_test = N_test

    @property
    def R(self):
        return self._R_train if self.training else self._R_test

    @property
    def N(self):
        return self._N_train if self.training else self._N_test

    def forward(self, xs, lengths, t):
        max_length, batch_size = xs.size()
        h = Variable(xs.data.new(1, batch_size, self.hidden_size).zero_().float(), requires_grad=False)
        state = (h, h)
        embs = functional.dropout(self.embed(xs), 0.1)
        rows = xs.data.new(batch_size).zero_()
        columns = xs.data.new(range(batch_size))
        log_probs = []
        baselines = []
        last_rows = rows.clone().fill_(max_length - 1)
        for _ in range(self.N):
            for _ in range(self.R):
                feed_previous = rows >= max_length
                rows = feed_previous.long() * last_rows + (1 - feed_previous.long()) * rows
                h, state = self._maybe_lstm(embs[rows, columns][None], state,
                                            feed_previous[None, :, None].expand_as(h))
                rows = rows + 1
                if self._finish_reading(rows, max_length):
                    break
            feed_previous = rows >= max_length
            # TODO: replace where function when it is added
            rows = feed_previous.long() * last_rows + (1 - feed_previous.long()) * rows
            h, state = self._maybe_lstm(embs[rows, columns][None], state,
                                        feed_previous[None, :, None].expand_as(h))
            p = functional.softmax(self.linear(h.squeeze(0)), dim=1)
            m = Categorical(p)
            jump = m.sample()
            log_prob = m.log_prob(jump)
            log_prob.data.masked_fill_(feed_previous, 0)
            log_probs.append(log_prob)
            baselines.append(self.baseline(h.squeeze(0)))
            feed_previous = (jump.data == 0).long()
            rows = feed_previous * (last_rows + 1) + (1 - feed_previous) * (rows + jump.data)
            if self._finish_reading(rows, max_length):
                break
        y = functional.log_softmax(self.output(h.squeeze(0)), dim=1)
        reward = self._get_reward(y, t)
        baseline = torch.cat(baselines, dim=1)
        return self.nll_loss(y, t) + self._reinforce(log_probs, reward, baselines) + \
            self.mse_loss(baseline, reward.expand_as(baseline))

    def inference(self, xs):
        max_length, batch_size = xs.size()
        h = Variable(xs.data.new(1, batch_size, self.hidden_size).zero_().float(), volatile=True)
        state = (h, h)
        embs = self.embed(xs)
        rows = xs.data.new(batch_size).zero_()
        columns = xs.data.new(range(batch_size))
        last_rows = rows.clone().fill_(max_length - 1)
        for _ in range(self.N):
            for _ in range(self.R):
                feed_previous = rows >= max_length
                rows = feed_previous.long() * last_rows + (1 - feed_previous.long()) * rows
                h, state = self._maybe_lstm(embs[rows, columns][None], state,
                                            feed_previous[None, :, None].expand_as(h))
                rows = rows + 1
                if self._finish_reading(rows, max_length):
                    break
            feed_previous = (rows >= max_length)
            # TODO: replace where function when it is added
            rows = feed_previous.long() * last_rows + (1 - feed_previous.long()) * rows
            h, state = self._maybe_lstm(embs[rows, columns][None], state,
                                        feed_previous[None, :, None].expand_as(h))
            p = functional.softmax(self.linear(h.squeeze(0)), dim=1)
            jump = p.max(dim=1)[1]
            feed_previous = (jump.data == 0).long()
            rows = feed_previous * (last_rows + 1) + (1 - feed_previous) * (rows + jump.data)
            if self._finish_reading(rows, max_length):
                break
        return self.output(h.squeeze(0)).max(dim=1)[1]

    @staticmethod
    def _finish_reading(rows, max_length):
        return (rows >= max_length).all()

    def _maybe_lstm(self, x, state, mask):
        h_new, state_new = self.lstm(x, state)
        c = state_new[1]
        # filling with 0
        # h_new.data.masked_fill_(mask, 0)
        # c.data.masked_fill_(mask, 0)
        # feeding previous state
        h_new.data.masked_scatter_(mask, state[0].data)
        c.data.masked_scatter_(mask, state[1].data)
        return h_new, (h_new, c)

    def _reinforce(self, log_probs, reward, baselines):
        if not log_probs:
            return Variable(baselines[0].data.new(1).zero_(), requires_grad=False)
        return torch.mean(torch.cat([- (reward - b) * l for l, b in zip(log_probs, baselines)], dim=1))

    def _get_reward(self, y, t):
        correct = y.data.max(dim=1)[1].eq(t.data).float()
        return Variable(correct.masked_fill_(correct == 0., -1), requires_grad=False).unsqueeze(1)
