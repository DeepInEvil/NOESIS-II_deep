from data_utils import UDC
from transformer_rnn import TransformerRNN
from args import get_args
from eval import eval_model
import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score
import torch.nn as nn

args = get_args()
if args.gpu:
    torch.cuda.manual_seed(args.randseed)
data = UDC(train_inp=args.train_inp,
           val_inp=args.val_inp)

model = TransformerRNN(emb_dim=args.input_size, n_vocab=data.bpe.vocab_size(), rnn_h_dim=256, gpu = args.gpu)
criteria = nn.NLLLoss()
solver = optim.Adam(model.parameters(), lr=args.lr)


def train():
    for epoch in range(args.epochs):
        model.train()
        print('\n\n-------------------------------------------')
        print('Epoch-{}'.format(epoch))
        print('-------------------------------------------')

        train_iter = enumerate(data.get_batches('train'))
        if not args.no_tqdm:
            train_iter = tqdm(train_iter)
            train_iter.set_description_str('Training')
            train_iter.total = len(data.train)

        for it, mb in train_iter:
            c, c_u_m, c_m, r, r_u_m, r_m, y = mb
            # print (c, c_u_m, c_m, r, y)
            # getting predictions
            pred = model(c, c_u_m, c_m, r, r_u_m, r_m)

            #train_iter.set_description(model.print_loss())

            #loss = F.nll_loss(pred, r)
            #loss = criteria(pred, y)
            #y = torch.argmax(y)
            #print (y.size())
            loss = criteria(pred, y)

            loss.backward()
            #print (model.conv3.grad)
            #clip_gradient_threshold(model, -10, 10)
            solver.step()
            solver.zero_grad()


        val_mrr = eval_model(model, data, 'valid')
        print ('Validation MRR for this epoch:'+str(val_mrr))


if __name__ == '__main__':
    train()