import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from Unet import UNet
from torch.utils.data import DataLoader
from dataloader import TrainDatase


class DenoisingGAN(nn.Module):
    def __init__(self, input_shape, output_shape, num_layers=2):
        super(DenoisingGAN, self).__init__()

        # Define the generator network (U-Net)
        self.generator = UNet(input_shape, output_shape, num_layers)

        # Define the discriminator network (a simple convolutional neural network)
        self.discriminator = nn.Sequential(
            nn.Conv2d(output_shape[0], 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1),
            nn.Sigmoid()
        )

        # Define the loss function
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Initialize the weights of the generator and discriminator networks
        self.generator.apply(self._weights_init)
        self.discriminator.apply(self._weights_init)

    def _weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Pass the input through the generator network to produce the fake image
        fake_image = self.generator(x)

        # Pass the fake image through the discriminator network to produce the predicted probability
        pred_prob = self.discriminator(fake_image)

        # Compute the loss between the predicted probability and the true label (which is noisy for real images and 0 for fake images)
        loss = self.loss_fn(pred_prob, torch.ones_like(pred_prob))

        return loss

    def train(self, dataset, epochs=10, batch_size=32, lr=0.001):
        # Create a dataloader from the dataset
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Set the model to training mode
        self.train()

        # Iterate over the training data in mini-batches
        for epoch in range(epochs):
            for i, (images, _) in enumerate(dataloader):
                # Zero the gradients
                self.zero_grad()

                # Forward pass
                loss = self(images)

                # Backward pass
                loss.backward()

                # Update the weights
                self.optimizer.step()

                # Print the loss at each iteration
                print('Epoch {}: Loss = {:.4f}'.format(epoch + 1, loss.item()))