import torch
import torchvision


import matplotlib.pyplot as plt

from vae import VAE


def main():
    # hyper parameters
    input_dim = 784
    hidden_dim = 200
    latent_dim = 20

    # load model
    model = VAE(input_dim, hidden_dim, latent_dim)
    model.load_state_dict(torch.load('./vae.pt'))

    with torch.no_grad():
        sample_size = 64
        z = torch.randn(sample_size, latent_dim)
        x = model.decorder(z)
        generated_images = x.view(sample_size, 1, 28, 28)



    grid_img = torchvision.utils.make_grid(
        generated_images,
        nrow=8,
        padding=2,
        normalize=True,
    )

    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis('off')
    plt.savefig('generated_images.png')





if __name__ == '__main__':
    main()
