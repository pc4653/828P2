from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms

batch_size = 64
epochs = 20
#torch.cuda.manual_seed(1)
log_interval = 10
kwargs = {'num_workers': 1, 'pin_memory': True} 

dset = np.load('data/vae-mutation-counts.npy')
#dset = np.load('data/vae_sim_70000-mutation-counts.npy')

#max_num = np.amax(dset)
#dset = dset/max_num

dset = np.split(dset, len(dset)/len(dset[0]))

for i in range(len(dset)):
    dset[i] = torch.from_numpy(dset[i])
    dset[i] = dset[i].float()

train_data = dset[0:9000]	
test_data = dset[9000:len(dset)]		

#train_data = dset[0:60000]	
#test_data = dset[60000:len(dset)]	
	
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)
        else:
          return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.relu(self.fc4(h3))
        #return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE()
model.cuda()


def loss_function(recon_x, x, mu, logvar):
    #BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784))
    MSE = F.mse_loss(recon_x, x.view(-1,784))
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= batch_size * 784
    #return BCE + KLD
    return MSE + KLD


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
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

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
    sample = Variable(torch.randn(64, 20))
    sample = sample.cuda()
    sample = model.decode(sample).cpu()
    sample = sample.data.view(64*28, 28)
    np.save('vae_results/sample_' + str(epoch) + '.npy', sample.numpy())