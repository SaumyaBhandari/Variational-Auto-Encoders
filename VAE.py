import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import torch.optim as optim
import numpy as np 
import torch.nn as nn
from dataset import DataLoader
import math
from tqdm import tqdm
from metric import MixedLoss
from tensorboardX import SummaryWriter 






class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out







class Encoder(nn.Module):

    def __init__(self, block, layers):

        self.inplanes = 64
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.conv2 = nn.Conv2d(512, 128, 3, 1, 1)
        self.conv3 = nn.Conv2d(128, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 16, 3, 1, 1)
        self.fc_mu = nn.Linear(16384, 16384)
        self.fc_var = nn.Linear(16384, 16384)
        self.dropout = nn.Dropout2d(0.3)



    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)



    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu



    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        mu = self.relu(self.fc_mu(x))
        sigma = self.relu(self.fc_var(x))
        x = self.reparameterize(mu, sigma)
        return x, mu, sigma






class Decoder(nn.Module):
    def __init__(self, outputDeterminer):
        super(Decoder, self).__init__()
        self.outputDeterminer = outputDeterminer
        self.pre_conv = nn.Conv2d(16, 64, 3, 1, 1)
        self.conv1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv2 = nn.Conv2d(128, 256, 3, 1, 1)
        self.dconv1 = nn.ConvTranspose2d(256, 256, 2, 2,padding = 0)
        self.dconv2 = nn.ConvTranspose2d(256, 384, 2, 2, padding = 0)
        self.dconv3 = nn.ConvTranspose2d(384, 192, 2, 2, padding = 0)
        self.conv3 = nn.Conv2d(192, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 8, 3, padding=1)
        self.relu = nn.ReLU()
        if self.outputDeterminer.lower() == "image":
            self.outputDeterminerConv = nn.Conv2d(8, 3, 3, padding=1)
            self.finalactivation = nn.ReLU()
        elif self.outputDeterminer.lower() == "mask":
            self.outputDeterminerConv = nn.Conv2d(8, 1, 3, padding=1)
            self.finalactivation = nn.Sigmoid()


    def forward(self,x):
        x = x.view(-1, 16, 32, 32)
        x = self.relu(self.conv2(self.relu(self.conv1(self.relu(self.pre_conv(x))))))
        x = self.relu(self.dconv1(x))
        x = self.relu(self.dconv2(x))
        x = self.relu(self.dconv3(x))
        x = self.relu(self.conv4(self.relu(self.conv3(x))))
        x = self.finalactivation(self.outputDeterminerConv(x))
        return x





class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.encoder = Encoder(Bottleneck, [3, 4])
        self.decoder = Decoder(outputDeterminer="image")

    def forward(self,x):
        latent, mu, sigma = self.encoder(x)
        decoded = self.decoder(latent) 
        return decoded, mu, sigma






def save_sample(epoch, x, img_pred):
    path = f'Training Sneakpeeks/VAE_Results/{epoch}'
    elements = [x, img_pred]
    elements = [transforms.ToPILImage()(torch.squeeze(element[0:1, :, :, :])) for element in elements]
    elements[0] = elements[0].save(f"{path}_image.jpg")
    elements[1] = elements[1].save(f"{path}_image_pred.jpg")




def train(epochs, batch_size=1, lr=0.0001):


    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    print(f"Using {device} device.")
    print("Loading Datasets...")
    train_dataloader = DataLoader().load_data(batch_size)
    print("Dataset Loaded.")
    print("Initializing Parameters...")


    model = Autoencoder()
    model = model.to(device)
    writer = SummaryWriter(log_dir="logs")  
    optimizerImg = optim.AdamW(model.parameters(), lr)
    nvidia_mix_loss = MixedLoss(0.5, 0.5)
    kld_weight = batch_size
    beta = 4
    loss_train = []
    start = 0
    epochs = epochs
    print(f"Parameters Initialized...")
    print(f"Starting to train for {epochs} epochs.")

    for epoch in range(start, epochs):
        print(f"Epoch no: {epoch+1}")
        _loss = 0
        

        for i, image in enumerate(tqdm(train_dataloader)):

            image = image.to(device)

            optimizerImg.zero_grad()

            image_pred, mu, sigma = model(image)

            loss_mix = nvidia_mix_loss(image_pred, image)
            kld_loss = torch.mean(-0.5 * torch.sum(1 + sigma - mu ** 2 - sigma.exp(), dim = 1), dim = 0)
            kld_loss = kld_loss*beta*kld_weight
            loss = loss_mix + kld_loss

            _loss += loss.item()
            loss.backward()
            optimizerImg.step()
            save_sample(epoch+1, image, image_pred)
        writer.add_scalar("Training Loss", _loss, epoch)
        loss_train.append(_loss)


        print(f"Epoch: {epoch+1}, Training loss: {_loss}")
        if loss_train[-1] == min(loss_train):

            print('Saving CNN Decoder Model...')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizerImg.state_dict(),
                'loss': loss_train
            }, f'saved_model/VAE.tar')



        print('\nProceeding to the next epoch...')



train(epochs=70)