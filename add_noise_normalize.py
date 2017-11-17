import numpy as np
mut = np.load('data/all_exomes.npy')
mut = np.transpose(mut).astype(float)
new_mut = np.zeros(new_mut.shape)
for i in range(len(mut)):
    noise = abs(np.random.normal(mut[i].mean()/50, mut[i].var()/50, mut[i].shape)).astype(float)
    new_mut[i] = mut[i] + noise
    new_mut[i] = mut[i]/max(mut[i])
    
return new_mut
