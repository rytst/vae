import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import datasets, transforms


from tqdm import tqdm
import matplotlib.pyplot as plt



class Encorder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.linear_mu = nn.Linear(hidden_dim, latent_dim)
        self.linear_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.linear(x)
        h = F.relu(h)
        mu = self.linear_mu(h)
        logvar = self.linear_logvar(h)
        sigma = torch.exp(0.5 * logvar)
        return mu, sigma



class Decorder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = self.linear1(z)
        h = F.relu(h)
        h = self.linear2(h)
        x_hat = F.sigmoid(h)
        return x_hat




def reparameterize(mu, sigma):
    eps = torch.randn_like(sigma)
    z = mu + eps * sigma
    return z




class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encorder = Encorder(input_dim, hidden_dim, latent_dim)
        self.decorder = Decorder(latent_dim, hidden_dim, input_dim)


    def get_loss(self, x):

        mu, sigma = self.encorder(x)
        z = reparameterize(mu, sigma)
        x_hat = self.decorder(z)

        batch_size = len(x)

        L1 = F.mse_loss(x_hat, x, reduction='sum')
        L2 = - torch.sum(1 + torch.log(sigma ** 2) - mu ** 2 - sigma ** 2)

        return (L1 + L2) / batch_size






def main():

    # hyper parameters
    input_dim = 784
    hidden_dim = 200
    latent_dim = 20
    epochs = 30
    learning_rate = 3e-4
    batch_size = 32

    # import dataset

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(torch.flatten)
    ])

    dataset = datasets.MNIST(
        root='./data',
        train=True,
        transform=transform,
        download=True,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )


    # model, optimizer
    model = VAE(input_dim, hidden_dim, latent_dim)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    epoch_history = []
    losses = []


    # training
    for epoch in tqdm(range(epochs)):
        loss_sum = 0.0
        cnt = 0

        for x, label in dataloader:
            optimizer.zero_grad()
            loss = model.get_loss(x)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            cnt += 1

        loss_avg = loss_sum / cnt

        epoch_history.append(epoch+1)
        losses.append(loss_avg)




    # plotting
    plt.plot(epoch_history, losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('loss.png')



    # save
    torch.save(model.state_dict(), './vae.pt')





if __name__ == '__main__':
    main()
