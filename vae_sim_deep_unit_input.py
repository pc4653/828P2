from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.utils.data
import random
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
#threshold = 1000000
batch_size = 25
test_size = 100
epochs = 10000
#torch.cuda.manual_seed(1)
log_interval = 10
kwargs = {'num_workers': 1, 'pin_memory': True} 

train_loss_log = []
dset = np.load('exomes_noise_normalize.npy')
#dset = np.transpose(dset)
#dset = np.load('data/noise_100_size_10000-mutation-counts.npy')
#dset = np.load('data/vae-mutation-counts.npy')
#dset = np.load('data/vae_sim_70000-mutation-counts.npy')

#max_num = np.amax(dset)
#dset = dset/max_num

#dset = np.split(dset, len(dset)/len(dset[0]))

test_ind = []
testdata = []
for i in range(test_size):
    j = random.randint(1, len(dset))
    test_ind.append(j)
    testdata.append(dset[j])
    
dset = np.delete(dset, test_ind, 0)    
testdata = np.asarray(testdata)

train_data = []
test_data = []
for i in range(len(dset)):
    temp = torch.from_numpy(dset[i].astype(float))
    temp = temp.float()
    train_data.append(temp)

for i in range(len(testdata)):
    temp = torch.from_numpy(testdata[i].astype(float))
    temp = temp.float()
    test_data.append(temp)    


#train_data = dset[0:60000]	
#test_data = dset[60000:len(dset)]	
	
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(96, 20)
        self.fc2 = nn.Linear(20, 20)		
        self.fc3 = nn.Linear(20, 20)		
        self.fc4 = nn.Linear(20, 20)	
        self.fc51 = nn.Linear(20, 10)
        self.fc52 = nn.Linear(20, 10)
        self.fc6 = nn.Linear(10, 20)		
        self.fc7 = nn.Linear(20, 20)
        self.fc8 = nn.Linear(20, 20)
        self.fc9 = nn.Linear(20, 20)
        self.fc10 = nn.Linear(20, 96)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h1 = self.relu(self.fc2(h1))
        h1 = self.relu(self.fc3(h1))
        h1 = self.relu(self.fc4(h1))
        return self.fc51(h1), self.fc52(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h1 = self.relu(self.fc6(z))
        h1 = self.relu(self.fc7(h1))
        h1 = self.relu(self.fc8(h1))
        h1 = self.relu(self.fc9(h1))
        return self.relu(self.fc10(h1))
        #return self.fc4(h3)

    def forward(self, x):
        #print(x)
        mu, logvar = self.encode(x.view(-1, 96))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE()
model.cuda()


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 96))
    #MSE = F.mse_loss(recon_x, x.view(-1,96))
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= batch_size * 96
    return BCE + KLD
    #return MSE + KLD


optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = Variable(data)
        data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        #if loss.data[0] > threshold:
        #    loss.data[0] = 100000
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))
            train_loss_log.append(train_loss)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    for i, data in enumerate(test_loader):
        data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).data[0]
        #if i == 0:
        #  n = min(data.size(0), 8)
        #  comparison = torch.cat([data[:n],recon_batch.view(batch_size, 1, 28, 28)[:n]])
        #  np.save('results/reconstruction_' + str(epoch) + '.npy',comparison.data.numpy())

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, epochs + 1):
    train(epoch)
    test(epoch)
    if epoch%10 == 0:
        sample = Variable(torch.randn(6000, 10))
        sample = sample.cuda()
        sample = model.decode(sample).cpu()
        sample = sample.data.view(6000, 96)
        np.save('vae_results/sample_' + str(epoch) + '.npy', sample.numpy())
        
import matplotlib.pyplot as plt        
plt.plot(np.asarray(train_loss_log, dtype = np.float32))
plt.show()



baseline = np.load('exomes_noise_normalize.npy')



