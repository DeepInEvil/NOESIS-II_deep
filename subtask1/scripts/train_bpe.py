import json
import youtokentome as yttm
#import os

#out_train_path = open('/home/debanjan/projects_deep/NOESIS-II_deep/subtask1/data/Task_1/ubuntu/train_bpe.txt', 'w') # Path to save train file for training bpe
model_path = ('/home/debanjan/projects_deep/NOESIS-II_deep/subtask1/data/Task_1/ubuntu/bpe_ubuntu.model')
#import random



def create_train_txt(train_in_f):
    # Read training file
    with open(train_in_f, 'r') as f:
        train = json.load(f)

    for t in train:
        utteranes = []
        for u in t['messages-so-far']:
            utteranes.append(u['utterance'])
        for u in t['options-for-next']:
            utteranes.append(u['utterance'])
        out_train_path.write(' '.join(u for u in utteranes) + '\n')


if __name__ == '__main__':
    #create_train_txt('/home/debanjan/projects_deep/NOESIS-II_deep/subtask1/data/Task_1/ubuntu/task-1.ubuntu.train.json')
    #out_train_path.close()
    # Training model
    train_data_path = '/home/debanjan/projects_deep/NOESIS-II_deep/subtask1/data/Task_1/ubuntu/train_bpe.txt'
    yttm.BPE.train(data=train_data_path, vocab_size=8000, model=model_path, n_threads=-1)

    # Loading model
    bpe = yttm.BPE(model=model_path)

    # Two types of tokenization
    print(bpe.encode(['how can i boost microphone volume? The volume is toooooo low'], output_type=yttm.OutputType.ID))
    print(bpe.encode(['how can i boost microphone volume? The volume is toooooo low'], output_type=yttm.OutputType.SUBWORD))
    print(bpe.encode(['try traceroute and do a ps -ef on it'], output_type=yttm.OutputType.SUBWORD))




