import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable

import heapq

from config import Config


class Sentence(object):

    def __init__(self, sentence, state, score, logprob):
        self.sentence = sentence
        self.state = state
        self.score = score
        self.logprob = logprob

    def __cmp__(self, other):
        assert isinstance(other, Sentence)
        if self.score == other.score:
            return 0
        elif self.score < other.score:
            return -1
        else:
            return 1


class TopMaintainer(object):

    def __init__(self, n):
        self.n = n
        self.data = []

    def size(self):
        return len(self.data)

    def push(self, x):
        if len(self.data) < self.n:
            heapq.heappush(self.data, x)
        else:
            heapq.heappushpop(self.data, x)

    def extract(self):
        data = self.data
        self.data = None
        return data

    def reset(self):
        self.data = []


class Caption(nn.Module):

    def __init__(self,
                 input_size=Config.dim_feature,
                 embedding_size=Config.dim_embedding,
                 dim_attention=Config.dim_attention,
                 hidden_size=Config.dim_hidden,
                 num_layers=Config.num_layers,
                 n_words=Config.n_words):

        # avoid using super() for some serious reason
        nn.Module.__init__(self)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.input = nn.Linear(input_size, embedding_size)
        self.embedding = nn.Embedding(n_words, embedding_size)

        # self.attention = nn.Sequencial(nn.Linear(), nn.ReLU(), nn.Linear())

        self.init_state_h = nn.Linear(input_size, hidden_size)
        self.init_state_c = nn.Linear(input_size, hidden_size)

        lstm_input_size = embedding_size if Config.visual_attention is False \
            else (embedding_size + dim_attention)

        # self.rnn = nn.LSTM(lstm_input_size, hidden_size, num_layers=num_layers)

        self.rnn_cell = nn.LSTMCell(lstm_input_size, hidden_size)

        self.dropout = nn.Dropout(0.4)

        self.output = nn.Linear(hidden_size, n_words)

        self.init_weights()

    def forward(self, features, seqs, lengths):

        batch_size = features.size(0)
        # print batch_size

        # (batch_size, embedding_size)
        embed_features = self.input(features)

        # (batch_size, max_seqlen, embedding_size)
        embeddings = self.embedding(seqs)

        # (batch_size, 1 + max_seqlen, embedding_size)
        # 'features' is fed into the LSTM as the initial input
        # at each following time step, a embedded represention
        #   of the last word is fed into the LSTM.
        embeddings = torch.cat((embed_features.unsqueeze(1), embeddings), 1)

        # dropout on LSTM input. embeddings: (batch_size, 1 + max_seqlen, embedding_size)
        embeddings = self.dropout(embeddings)
        # packed embeddings -> contains image feature and embedded words

        # LSTM initial states
        if Config.lstm_init_state:
            lstm_state_h = self.init_state_h(self.dropout(features))
            lstm_state_h = nn.functional.tanh(lstm_state_h)  # (batch_size, hidden_size)
            lstm_state_c = self.init_state_h(self.dropout(features))
            lstm_state_c = nn.functional.tanh(lstm_state_c)  # (batch_size, hidden_size)

        else:
            lstm_state_h = Variable(torch.zeros(batch_size, self.hidden_size))
            lstm_state_c = Variable(torch.zeros(batch_size, self.hidden_size))

            lstm_state_h = lstm_state_h.cuda() if Config.use_cuda else lstm_state_h
            lstm_state_c = lstm_state_c.cuda() if Config.use_cuda else lstm_state_c

        state = (lstm_state_h, lstm_state_c)

        # all hidden outputs, sth. like (1, batch_size, hidden_size)
        # hiddens, _ = self.rnn(packed, (lstm_state_h, lstm_state_c))

        hiddens = Variable(torch.zeros(Config.maxlen, batch_size, self.hidden_size))
        hiddens = hiddens.cuda() if Config.use_cuda else hiddens

        # run LSTM forwarding
        for t in range(Config.maxlen):
            state = self.rnn_cell(embeddings[:, t, :], state)
            hiddens[t] = state[0]

        # hiddens -> (batch_size, maxlen, n_words)
        hiddens = torch.transpose(hiddens.squeeze(1), 0, 1)

        # dropout on LSTM output
        # hiddens[0]: (batch_size, length, hidden_size)
        outputs = self.dropout(hiddens)

        # (batch_size, length, n_words)
        outputs = self.output(outputs)
        return outputs

    def init_weights(self):
        v = 0.01
        self.input.weight.data.uniform_(-v, v)
        self.input.bias.data.fill_(0)

        self.embedding.weight.data.uniform_(-v, v)

        self.output.weight.data.uniform_(-v, v)
        self.output.bias.data.fill_(0)

        self.init_state_h.weight.data.uniform_(-v, v)
        self.init_state_h.bias.data.fill_(0)

        self.init_state_c.weight.data.uniform_(-v, v)
        self.init_state_c.bias.data.fill_(0)

        nn.init.uniform(self.rnn_cell.weight_ih, -v, v)
        nn.init.uniform(self.rnn_cell.weight_hh, -v, v)
        nn.init.uniform(self.rnn_cell.bias_ih, -v, v)
        nn.init.uniform(self.rnn_cell.bias_hh, -v, v)

        # orthogonal initialization
        if Config.orthogonal:
            nn.init.orthogonal(self.rnn_cell.weight_hh[0: self.hidden_size])
            nn.init.orthogonal(self.rnn_cell.weight_hh[self.hidden_size: 2 * self.hidden_size])
            nn.init.orthogonal(self.rnn_cell.weight_hh[2 * self.hidden_size: 3 * self.hidden_size])
            nn.init.orthogonal(self.rnn_cell.weight_hh[3 * self.hidden_size: 4 * self.hidden_size])

        # print shapes
        # print 'weight_ih_l0', self.rnn.weight_ih_l0.shape
        # print 'weight_hh_l0', self.rnn.weight_hh_l0.shape
        # print 'bias_ih_l0', self.rnn.bias_ih_l0.shape  # this is forget gate?
        # print 'bias_hh_l0', self.rnn.bias_hh_l0.shape

        # Jozefowicz, R., Zaremba, W., & Sutskever, I. (2015).
        #  An empirical exploration of recurrent network architectures.
        #  In Proceedings of the 32nd International Conference on Machine Learning (ICML-15)
        #  (pp. 2342-2350).
        # self.rnn.bias_ih_l0.data[self.hidden_size:2 * self.hidden_size].uniform_(0.9, 1.1)

    def beam_search(self, feature):
        """inference with beam search

        Input:
            features: image feature with shape (1536,).
        Output:
            sentences: inferenced sentences, list of word_id`s.
        """

        # function for one-step LSTM
        def get_topk_words(embed, states):
            # embed (batch_size, embed_size (== input_size))
            # states tuple (h, c), both are (batch_size, hidden_size)
            states = self.rnn_cell(embed, states)
            # print 'input', embed.size(), 'states', states[0].size(), states[1].size()

            # output (batch_size, n_words)
            output = self.output(states[0])
            # print 'output', output.size()

            logprobs = torch.nn.functional.log_softmax(output, dim=1)
            # print 'after softmax', logprobs.size()
            logprobs, words = logprobs.topk(Config.beam_size, dim=1)
            # print 'topk', logprobs.size(), words.size()
            return words.data, logprobs.data, states

        # embedding image feature
        input = self.input(feature)
        input = input.unsqueeze(0)  # (1, embed_size)

        if Config.lstm_init_state:
            # initialize states
            feature = feature.unsqueeze(0)  # (1, input_size)
            lstm_state_h = self.init_state_h(self.dropout(feature))
            lstm_state_h = nn.functional.tanh(lstm_state_h)  # (1, hidden_size)
            lstm_state_c = self.init_state_h(self.dropout(feature))
            lstm_state_c = nn.functional.tanh(lstm_state_c)  # (1, hidden_size)
            # print lstm_state_c.shape
            # print lstm_state_h.shape

            state = (lstm_state_h, lstm_state_c)
        else:
            state = (Variable(torch.zeros(1, self.hidden_size)),
                     Variable(torch.zeros(1, self.hidden_size)))
        if Config.use_cuda:
            state = [s.cuda() for s in state]

        sentences = TopMaintainer(Config.beam_size)
        answers = TopMaintainer(Config.beam_size)

        # initialize sentences
        words, logprobs, new_state = get_topk_words(input, state)
        for k in range(Config.beam_size):
            sentence = Sentence(
                sentence=[words[0, k]],
                state=new_state,
                logprob=logprobs[0, k],
                score=logprobs[0, k])
            sentences.push(sentence)

        # beam search
        for _ in range(Config.maxlen - 1):
            sentence_list = sentences.extract()
            sentences.reset()
            input_feed = torch.LongTensor([c.sentence[-1] for c in sentence_list])
            input_feed = input_feed.cuda() if Config.use_cuda else input_feed
            input_feed = Variable(input_feed, volatile=True)

            # print 'input_feed constructed', input_feed.size()

            state_feed = [c.state for c in sentence_list]

            state_feed_h, state_feed_c = zip(*state_feed)
            state_feed = (torch.cat(state_feed_h, 0),
                          torch.cat(state_feed_c, 0))

            embed = self.embedding(input_feed)

            words, logprobs, new_states = get_topk_words(embed, state_feed)

            for i, p_sentence in enumerate(sentence_list):
                state = (new_states[0].narrow(0, i, 1),
                         new_states[1].narrow(0, i, 1))

                for k in range(Config.beam_size):
                    sentence = p_sentence.sentence + [words[i, k]]
                    logprob = p_sentence.logprob + logprobs[i, k]

                    #  Note that this metric is controversial.
                    score = logprob
                    # score = logprob / len(sentence)

                    if words[i, k] == 0:  # <eos>
                        beam = Sentence(sentence, state, score, logprob)
                        answers.push(beam)
                    else:
                        beam = Sentence(sentence, state, score, logprob)
                        sentences.push(beam)
            if sentences.size() == 0:
                break

        if answers.size() == 0:
            answers = sentences

        answer = answers.extract()
        answer.sort(reverse=True)

        return [c.sentence for c in answer], [c.score for c in answer]

    def inference(self, feature):
        sentences, scores = self.beam_search(feature)
        ans = sentences[0]

        if ans[-1] == 0:  # cut off <eos>
            ans = ans[:-1]
        return ans
