import torch
from torch import nn
from torch.nn import functional
from torch.autograd import Variable
from torch.distributions import Categorical


class LSTMJump(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, categories,
                 N, K, R):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size)
        self.linear = nn.Linear(hidden_size, K + 1)
        self.output = nn.Linear(hidden_size, categories)
        self.baseline = nn.Linear(hidden_size, 1)
        self.nll_loss = nn.NLLLoss()
        self.mse_loss = nn.MSELoss(size_average=False)
        self.hidden_size = hidden_size
        self.N = N
        self.R = R

    def forward(self, xs, t):
        max_length, batch_size = xs.size()
        h = xs.data.new(1, batch_size, self.hidden_size).zero_()
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
        y = functional.log_softmax(self.output(h.squeeze(0)), dim=1)
        reward = self._get_reward(y, t)
        baseline = torch.cat(baselines, dim=1)
        return self.nll_loss(y, t) + self._reinforce(log_probs, reward, baselines) + \
            self.mse_loss(baseline, reward.expand_as(baseline))

    def _maybe_lstm(self, x, state, mask):
        print(x.size(), state[0].size(), mask.size())
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

    def inference(self, xs):
        state = None
        max_length, batch_size = xs.size()
        embs = self.embed(xs)
        rows = xs.data.new(batch_size).zero_()
        columns = xs.data.new(range(batch_size))
        # reading
        length_read = self.R if self.R < max_length else max_length
        for _ in range(length_read):
            h, state = self.lstm(embs[rows, columns][None], state)
            rows = rows + 1
        # jump reading
        last_rows = rows.clone().fill_(max_length - 1)
        length_jump_read = self.N if self.R < max_length else 0
        for _ in range(length_jump_read):
            feed_previous = (rows >= max_length)
            # TODO: replace where function when it is added
            rows = feed_previous.long() * last_rows + (1 - feed_previous.long()) * rows
            h, state = self._maybe_lstm(embs[rows, columns][None], state,
                                        feed_previous[None, :, None].expand_as(h))
            p = functional.softmax(self.linear(h.squeeze(0)), dim=1)
            jump = p.max(dim=1)[1]
            rows = rows + jump.data + 1
        return self.output(h.squeeze(0)).max(dim=1)[1]
