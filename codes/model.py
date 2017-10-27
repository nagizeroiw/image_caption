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
                 hidden_size=Config.dim_hidden,
                 num_layers=Config.num_layers,
                 n_words=Config.n_words):

        # avoid using super() for some serious reason
        nn.Module.__init__(self)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.input = nn.Linear(input_size, embedding_size)
        self.embedding = nn.Embedding(n_words, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers)
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(hidden_size, n_words)
        self.init_weights()

    def forward(self, features, seqs, lengths):

        # (batch_size, embedding_size)
        features = self.input(features)

        # (batch_size, max_seqlen, embedding_size)
        embeddings = self.embedding(seqs)

        # (batch_size, 1 + max_seqlen, embedding_size)
        # 'features' is fed into the LSTM as the initial input
        # at each following time step, a embedded represention
        #   of the last word is fed into the LSTM.
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)

        # packed embeddings -> contains image feature and embedded words
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)

        # all hidden outputs, sth. like (1, batch_size, hidden_size)
        hiddens, _ = self.rnn(packed)

        # dropout at hidden-> output
        outputs = self.dropout(hiddens[0])

        # sth. like (batch_size, n_words)
        outputs = self.output(outputs)
        return outputs

    def init_weights(self):
        self.input.weight.data.uniform_(-0.1, 0.1)
        self.input.bias.data.fill_(0)
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.output.weight.data.uniform_(-0.1, 0.1)
        self.output.bias.data.fill_(0)
        # RNN init_weights?

    def beam_search(self, feature):
        """inference with beam search

        Input:
            features: image feature with shape (1536,).
        Output:
            sentences: inferenced sentences, list of word_id`s.
        """

        # function for one-step LSTM
        def get_topk_words(embed, states):
            hidden, new_states = self.rnn(embed, states)
            output = self.output(hidden.squeeze(0))
            logprobs = torch.nn.functional.log_softmax(output, dim=1)
            logprobs, words = logprobs.topk(Config.beam_size, dim=1)
            return words.data, logprobs.data, new_states

        # embedding image feature
        input = self.input(feature)
        input = input.unsqueeze(0).unsqueeze(0)  # (1, 1, embed_size)

        state = (Variable(torch.zeros(self.num_layers, 1, self.hidden_size)),
                 Variable(torch.zeros(self.num_layers, 1, self.hidden_size)))
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
            state_feed = [c.state for c in sentence_list]

            state_feed_h, state_feed_c = zip(*state_feed)
            state_feed = (torch.cat(state_feed_h, 1),
                          torch.cat(state_feed_c, 1))

            # (1, 5, 512)
            # print 'before feed', state_feed[0].shape

            embed = self.embedding(input_feed).view(1, len(input_feed), -1)
            words, logprobs, new_states = get_topk_words(embed, state_feed)

            for i, p_sentence in enumerate(sentence_list):
                state = (new_states[0].narrow(1, i, 1),
                         new_states[0].narrow(1, i, 1))

                # (1, 1, 512)
                # print 'ready to save', state[0].shape

                for k in range(Config.beam_size):
                    sentence = p_sentence.sentence + [words[i, k]]
                    logprob = p_sentence.logprob + logprobs[i, k]

                    # Scores are given by mean log-likelihood`s of all words.
                    #  Note that this metric is controversial.
                    score = logprob / len(sentence)

                    if words[i, k] == 0:  # <eos>
                        beam = Sentence(sentence, state, score, logprob)
                        answers.push(beam)
                    else:
                        beam = Sentence(sentence, state, score, logprob)
                        sentences.push(beam)
            if sentences.size() == 0:
                break

        if not answers.size():
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
