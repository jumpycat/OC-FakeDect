import torch
import os
from skimage import io, transform
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn.modules.module import _addindent
import numpy as np
import re
from tqdm import tqdm
from scipy.interpolate import interp1d
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, mean_absolute_error, classification_report, roc_curve, roc_auc_score
from torch.utils.data import DataLoader

batch_size = 128
epochs = 2000
no_cuda = False
seed = 1
log_interval = 50

cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)
device = torch.device("cuda" if cuda else "cpu")
print(device)

kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}

# HIGH res
train_root = 'data/deepfake_bgr/train/'

TRANSFORM_IMG = transforms.Compose([
    transforms.Resize(100),
    # transforms.CenterCrop(100),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
TRANSFORM_IMG_TEST = transforms.Compose([
    transforms.Resize(100),
    # transforms.CenterCrop(100),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(train_root, transform=TRANSFORM_IMG),
    batch_size=batch_size, shuffle=True)

# for evaluation/testing
def mse_loss_cal(input, target, avg_batch=True):
    ret = torch.mean((input - target) ** 2)
    return ret.item() 


class VAE_CNN(nn.Module):
    def __init__(self):
        super(VAE_CNN, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3,
                               stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 16, kernel_size=3,
                               stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)

        #self.drop = nn.Dropout(0.2)

        # Latent vectors mu and sigma
        self.fc1 = nn.Linear(25 * 25 * 16, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)
        self.fc21 = nn.Linear(1024, 1024)
        self.fc22 = nn.Linear(1024, 1024)

        # Sampling vector
        self.fc3 = nn.Linear(1024, 1024)
        self.fc_bn3 = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 25 * 25 * 16)
        self.fc_bn4 = nn.BatchNorm1d(25 * 25 * 16)

        self.relu = nn.ReLU()

        # Decoder

        self.conv5 = nn.ConvTranspose2d(
            16, 64, kernel_size=3, stride=1, padding=1,   bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(32)
        self.conv7 = nn.ConvTranspose2d(
            32, 16, kernel_size=3, stride=1, padding=1,   bias=False)
        self.bn7 = nn.BatchNorm2d(16)
        self.conv8 = nn.ConvTranspose2d(
            16, 3, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)

    def encode(self, x):
        c1 = self.bn1(self.conv1(x))
        conv1 = self.relu(c1)
        c2 = self.bn2(self.conv2(conv1))
        conv2 = self.relu(c2)
        c3 = self.bn3(self.conv3(conv2))
        conv3 = self.relu(c3)
        c4= self.bn4(self.conv4(conv3))
        conv4 = self.relu(c4)
        conv4 = conv4.view(-1, 25 * 25 * 16)

        fc1 = self.relu(self.fc_bn1(self.fc1(conv4)))
        r1 = self.fc21(fc1)
        r2 = self.fc22(fc1)
        return r1, r2

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.50).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def decode(self, z):
        #print(z.shape)
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        #print(fc3.shape)
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3)))
        #print(fc4.shape)
        fc4 = fc4.view(-1, 16, 25, 25)

        conv5 = self.relu(self.bn5(self.conv5(fc4)))
        #print(conv5.shape)
        conv6 = self.relu(self.bn6(self.conv6(conv5)))
        #print(conv6.shape)
        conv7 = self.relu(self.bn7(self.conv7(conv6)))
        #print(conv7.shape)
        conv8 = self.conv8(conv7)
        #print(conv8.shape)
        return conv8.view(-1, 3, 100, 100)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class customLoss(nn.Module):
    def __init__(self):
        super(customLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self.mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return loss_MSE + loss_KLD


model = VAE_CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_mse = customLoss()

train_losses = []


#ckpt = torch.load("dfdc/vae_pytorch_dfdc_FT_.pt")
# model.load_state_dict(ckpt)
#model = model.to(device)


for epoch in range(1, epochs + 1):
    # train(epoch)

    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        permute = [2, 1, 0]
        data = data[:, permute, :, :]

        recon_batch, mu, logvar = model(data)
        #mu1, logvar1 = model.encode(recon_batch)
        #z1 = model.get_hidden(mu1, logvar1)

        #loss = loss_mse(z, z1, mu, logvar)
        loss = loss_mse(recon_batch, data, mu, logvar)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                128. * batch_idx / len(train_loader),
                loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))
    train_losses.append(train_loss / len(train_loader.dataset))

    # EVALUATE YOUR MODEL HERE 
    # model.eval()
    # with torch.no_grad():


plt.figure(figsize=(15, 10))
plt.plot(range(len(train_losses[1:])), train_losses[1:], c="dodgerblue")
plt.title("Loss per epoch", fontsize=18)
plt.xlabel("epoch", fontsize=18)
plt.ylabel("loss", fontsize=18)
plt.legend(['Train. Loss'], fontsize=18)
plt.show()
