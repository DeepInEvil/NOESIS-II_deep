import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerRNN(nn.Module):
    def __init__(self, emb_dim, n_vocab, rnn_h_dim, gpu=False, emb_drop=0.3, pad_idx=0,
                 lstm_drop=0.3, transformer_dim=512, transformer_head=1, requires_grad=False):
        super(TransformerRNN, self).__init__()
        self.emb_dim = emb_dim
        self.h_dim = rnn_h_dim
        self.gpu = gpu
        self.emb_drop = nn.Dropout(emb_drop)
        self.rnn_drop = lstm_drop

        self.word_embed = nn.Embedding(n_vocab, emb_dim, padding_idx=pad_idx)
        self.context_rnn = nn.LSTM(input_size=self.emb_dim,
                                   hidden_size=self.h_dim,
                                   dropout=self.rnn_drop,
                                   bidirectional=True,
                                   batch_first=True)
        self.context_transformer = nn.TransformerEncoderLayer(d_model=self.h_dim*2, nhead=4)

        self.response_rnn = nn.LSTM(input_size=self.emb_dim,
                                   hidden_size=self.h_dim,
                                   dropout=self.rnn_drop,
                                   bidirectional=True,
                                   batch_first=True)

        self.M = nn.Linear(2*self.h_dim, 2*self.h_dim)
        #self.b = nn.Parameter(torch.FloatTensor([0]))

        if gpu:
            self.cuda()

    def forward(self, c, c_u_m, c_m, r, r_u_m, r_m):
        """

        :param c: Context Utterance SIZE (C) X S
        :param c_u_m: Context Utterance SIZE X S
        :param c_m: Context Utterance SIZE
        :param r: Response Utterance SIZE (R) X S
        :param r_u_m: Response Utterance SIZE X S
        :param r_m: Response Utterance SIZE
        :return: Response Utterance SIZE
        """

        c = self.emb_drop(self.word_embed(c))  # C X S X E
        r = self.emb_drop(self.word_embed(r))  # R X S X E

        c_out, (ht, ct) = self.context_rnn(c)  #
        c_out = self.concat_rnn_states(c_out, c_u_m)
        c_out = self.context_transformer(c_out) # pass through the transformer C X S X 2*H

        r_out, (ht, ct) = self.context_rnn(r)  #
        r_out = self.concat_rnn_states(r_out, r_u_m)[:, -1].squeeze()
        #c_out = F.max_pool1d(c_out, c_out.size(0))
        c_out = c_out[:, -1].squeeze().expand(r_out.size(0), c_out.size(0), c_out.size(-1))  # R X C X 2*H
        #c_out = F.max_pool1d(c_out)
        o = self.M(c_out.sum(1)).unsqueeze(1)  # R X 1 X 2*H
        # print (o.size(), r_out.size())
        o = torch.bmm(o, r_out.unsqueeze(2)).squeeze()
        # print (o.size())
        return F.log_softmax(o).unsqueeze(0) # unsqueeze for nll loss

    def concat_rnn_states(self, x, m):
        """
        concat forward and backward hidden states
        :param x: B X S X H * 2
        :param m: B X S X H
        :return: B X S X
        """
        x = x.view(x.size(0), x.size(1), 2, self.h_dim)
        fw = x[:, :, 0, :].squeeze() * m.unsqueeze(-1)
        bw = x[:, :, 1, :].squeeze() * m.unsqueeze(-1)
        H = torch.cat([fw, bw], dim=-1)

        return H