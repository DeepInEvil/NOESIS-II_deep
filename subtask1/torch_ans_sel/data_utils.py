import json
import youtokentome as yttm
import numpy as np
import torch
from args import get_args

arg = get_args()


class UDC:
    def __init__(self, train_inp, val_inp, test_inp=None):
        print ('Loading all data...........................')
        #load bpe model
        self.bpe = yttm.BPE(arg.bpe_model)
        #load main data
        #with open(train_inp, 'r') as f:
        #    self.train_in = json.load(f)
        with open(val_inp, 'r') as f:
            self.val_in = json.load(f)
        if test_inp:
            with open(test_inp, 'r') as f:
                self.test_in = json.load(f)

        print ('Loaded all data from DISK...')

        #self.train = self.process_data(self.train_in)
        self.valid = self.process_data(self.val_in)

        if test_inp:
            self.test = self.process_data(self.test_in)

    def process_data(self, inp_dat):
        """preprocess the data using bpe"""
        dat = []
        for t in inp_dat:
            context, responses = [], []
            # get input contexts
            for c in t['messages-so-far']:
                context_utterance = self.bpe.encode(c['utterance'], output_type=yttm.OutputType.ID)
                context.append(context_utterance)
            # get all responses
            for r in t['options-for-next']:
                resp_utterance = self.bpe.encode(r['utterance'], output_type=yttm.OutputType.ID)
                responses.append(resp_utterance)
            # correct response
            # print (t['options-for-next'])

            try:
                # print(t['options-for-correct-answers'][0]['candidate-id'])
                correct_resp = [i for i, r in enumerate(t['options-for-next'])
                            if t['options-for-correct-answers'][0]['candidate-id'] == r['candidate-id']]
                if len(correct_resp) > 1:
                    for r in correct_resp:
                        dat.append([context, responses, len(context), len(responses), list(r)])
                else:
                    dat.append([context, responses, len(context), len(responses), correct_resp])
                    #dat.append([context, responses, len(context), len(responses), correct_resp])
            except IndexError:
                # print(t['options-for-correct-answers'])
                continue
            # dat.append([context, responses, len(context), len(responses), correct_resp])

        return dat

    def get_batches(self, dataset='train'):
        # get iterations.
        #self.batch_size = batch_size
        if dataset == 'train':
            dataset = self.train
        elif dataset == 'valid':
            dataset = self.valid
            # print(dataset['team'])
        elif dataset == 'test':
            dataset = self.test

        for d in dataset:
            c, r, l_c, l_r, true_r = d
            c, c_u_m, c_m, r, r_u_m, r_m, y = self._load_batches(c, r, l_c, l_r, true_r)
            yield c, c_u_m, c_m, r, r_u_m, r_m, y

    @staticmethod
    def _load_batches(c, r, l_c, l_r, true_r):
        max_len_c = np.max([len(sent) for sent in c])
        max_len_r = np.max([len(sent) for sent in r])
        c_out = torch.zeros([l_c, max_len_c]).long()
        c_u_m = torch.zeros([l_c, max_len_c]).float()
        c_m = torch.zeros(l_c).float()

        r_out = torch.zeros([l_r, max_len_r]).long()
        r_u_m = torch.zeros([l_r, max_len_r]).float()
        r_m = torch.zeros(l_r).float()
        #y = torch.zeros(l_r)
        y = torch.tensor(true_r)  # for softmax loss

        for j, (row_c) in enumerate(c):
            c_out[j][:len(row_c)] = torch.Tensor(row_c)
            c_u_m[j][:len(row_c)] = 1
        c_m[:len(c)] = 1

        for j, (row_r) in enumerate(r):
            r_out[j][:len(row_r)] = torch.Tensor(row_r)
            r_u_m[j][:len(row_r)] = 1
        r_m[:len(r)] = 1

        # uncomment for sigmoid loss
        #for t in true_r:
        #y[true_r] = 1

        if arg.gpu:
            c_out, c_u_m, c_m, r_out, r_u_m, r_m, y = c_out.cuda(), c_u_m.cuda(), c_m.cuda(), r_out.cuda(), r_u_m.cuda(), \
                                              r_m.cuda(), y.cuda()

        return c_out, c_u_m, c_m, r_out, r_u_m, r_m, y


if __name__ == '__main__':
    data = UDC(train_inp=arg.train_inp,
               val_inp=arg.val_inp)
    train_iter = enumerate(data.get_batches('valid'))
    for it, mb in train_iter:
        c, c_u_m, c_m, r, r_u_m, r_m, y = mb




