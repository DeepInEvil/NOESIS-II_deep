import torch
import torch.nn.functional as F
import numpy as np
import scipy.stats as st
from tqdm import tqdm


def recall_at_k_np(scores, ks=[1, 2, 3, 4, 5]):
    """
    Evaluation recalll
    :param scores:  sigmoid scores
    :param ks:
    :return:
    """
    # sort the scores
    sorted_idxs = np.argsort(-scores, axis=1)
    ranks = (sorted_idxs == 0).argmax(1)
    recalls = [np.mean(ranks + 1 <= k) for k in ks]
    return recalls


def get_mrr(scores, y):
    "get MRR per batch"
    sorted_scores, indices = torch.sort(scores, descending=True)
    #pos = (indices == torch.argmax(y).item()).nonzero().item() # uncomment if sigmoid
    pos = (indices == y.item()).nonzero().item()
    return 1/(pos+1)


def eval_model(model, dataset, mode='valid', gpu=False, no_tqdm=False):
    """
    evaluation for DKE-GRU and AddGRU
    :param model:
    :param dataset:
    :param mode:
    :param gpu:
    :param no_tqdm:
    :return:
    """
    model.eval()
    mrr_scores = []

    assert mode in ['valid', 'test']

    data_iter = dataset.get_batches(mode)

    if not no_tqdm:
        data_iter = tqdm(data_iter)
        data_iter.set_description_str('Evaluation')
        n_data = len(dataset.valid) if mode == 'valid' else len(dataset.test)
        data_iter.total = n_data

    for mb in data_iter:
        c, c_u_m, c_m, r, r_u_m, r_m, y = mb

        # Get scores
        #scores_mb = torch.sigmoid(model(c, c_u_m, c_m, r, r_u_m, r_m))#Appropritate this line while running different models.
        scores_mb = (model(c, c_u_m, c_m, r, r_u_m, r_m))#Appropritate this line while running different models.
        # scores_mb = scores_mb.cpu() if gpu else scores_mb
        mrr_scores.append(get_mrr(scores_mb, y))

    # scores = np.concatenate(scores)

    # Handle the case when numb. of data not divisible by 10
    # mod = scores.shape[0] % 10
    # scores = scores[:-mod if mod != 0 else None]

    # scores = scores.reshape(-1, 10)  # 1 in 10
    # recall_at_ks = [r for r in recall_at_k_np(scores)]
    mrr_tot = np.average(mrr_scores)

    return mrr_tot