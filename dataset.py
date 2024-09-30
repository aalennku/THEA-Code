import torch
import numpy as np
from utils import PROFILE_TOKEN, PROFILE_VOCAB_SIZE, random_profile

# For training the encoder and decoder

class Random_DNA_dataset(torch.utils.data.Dataset):
    def __init__(self, N, length_dna, seed=None):
        super(Random_DNA_dataset, self).__init__()
        self.N = N
        self.length_dna = length_dna
        self.seed = seed
    def __len__(self):
        return self.N
    def __getitem__(self,idx):
        if not self.seed is None:
            np.random.seed(self.seed+idx)
        data = np.random.randint(0,4,size=(self.length_dna))
        if (not self.seed is None) and (idx % 1000==0):
            print(idx, data[0])
        return torch.tensor(data)
    
def generate_profile_batch(batch_size,length,padded_length,i,d,s,ratio,ratio_mode,max_err):
    profile_batch = []
    for _ in range(batch_size):
        profile = random_profile(length,i=i,d=d,s=s,ratio=ratio,ratio_mode=ratio_mode,max_err=max_err)
        for _ in range(padded_length-len(profile)):
            profile.append('<PAD>')
        profile = [PROFILE_TOKEN[_] for _ in profile]
        profile_batch.append(profile)
    return torch.tensor(profile_batch).T
