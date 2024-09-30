import numpy as np

# dna_token = {0:0,1:1,2:2,3:3,'<PAD>':6}
DNA_TOKEN = {_:_ for _ in range(4)}
DNA_TOKEN['<PAD>'] = 4
# profile_token: 0:no action; 1-3: subs; 4-7: inss; 8:del
PROFILE_TOKEN = {_:_ for _ in range(9)}
PROFILE_TOKEN['<PAD>'] = 9
DNA_VOCAB_SIZE = len(DNA_TOKEN)
PROFILE_VOCAB_SIZE = len(PROFILE_TOKEN)

def np_one_hot(s, num_classes=None):
    if num_classes is None:
        num_classes = s.max()+1
    b = np.zeros((*s.shape, num_classes))
    b[np.arange(s.size), s] = 1.
    return b

def ids_channel(s, profile):
    # ids_channel on string s
    ss = []
    idx = 0
    profile = profile[:]
    idx_profile = 0
    while idx_profile < len(profile):
        e = profile[idx_profile]
        if idx >= len(s):
            if e<=7 and e>3:
                ss.append(e%4)
            else:
                break
        else:
            if e <= 3:
                aligo = s[idx]
                aligo = (aligo + e) % 4
                ss.append(aligo)
                idx += 1
            elif e<=7:
                ss.append(e%4)
            elif e == 8:
                idx += 1
        idx_profile += 1
    return ss

def ids_channel_OH(s, profile):
    # ids_channel on string s in one hot format
    ss = []
    idx = 0
    profile = profile[:]
    idx_profile = 0
    while idx_profile < len(profile):
        e = profile[idx_profile]
        if idx >= len(s):
            if e<=7 and e>3:
                ss.append(np_one_hot(np.array([e%4]),4).tolist()[0])
            else:
                break
        else:
            if e <= 3:
                aligo = np.array(s[idx])
                aligo = np.roll(aligo,e)
                ss.append(aligo.tolist())
                idx += 1
            elif e<=7:
                ss.append(np_one_hot(np.array([e%4]),4).tolist()[0])
            elif e == 8:
                idx += 1
        idx_profile += 1
    return np.array(ss)

def random_profile(length,i,d,s,ratio,max_err=None,ratio_mode=None):
    profile = np.zeros(length*2).astype(int).tolist()
    random_vec = np.random.rand(length)
    if max_err is not None:
        positions = np.argwhere(random_vec<=ratio).flatten()
        if positions.shape[0] > max_err:
            positions = np.random.choice(positions,max_err)
            random_vec_mask = np.ones(length)
            for item in positions:
                random_vec_mask[item] = 0
            random_vec += random_vec_mask
    random_vec = random_vec.tolist()
    
    ratio_vec = np.zeros_like(random_vec) + ratio
    if ratio_mode == "ASC":
        ratio_vec *= np.array(np.arange(ratio_vec.shape[0]))*2/((ratio_vec.shape[0]-1))
    if ratio_mode == "DES":
        ratio_vec *= np.array(np.arange(ratio_vec.shape[0])[::-1])*2/((ratio_vec.shape[0]-1))
        
    ids = float(i+d+s)
    for idx, (random_number, ratio) in enumerate(zip(random_vec,ratio_vec)):
        if random_number <= ratio*(s)/ids:
            profile[idx] = np.random.randint(1,4)
        elif random_number <= ratio*(i+s)/ids:
            profile[idx] = np.random.randint(4,8)
        elif random_number <= ratio:
            profile[idx] = 8
    s_idx = 0
    for idx,e in enumerate(profile):
        if s_idx >= length:
            if e<=7 and e>3:
                break
            else:
                idx = idx-1
                break
        if e <= 3:
            s_idx += 1
        elif e <= 7:
            pass
        elif e == 8:
            s_idx += 1
    profile = profile[:idx+1]
    return profile