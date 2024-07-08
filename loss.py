import torch
from torch import nn
from torchvision.models import vgg19


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        model = vgg19(pretrained=True)
        self.vgg = model.features[:18].eval()  # Use the features up to the 18th layer
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        input_vgg = self.vgg(input)
        target_vgg = self.vgg(target)
        loss = nn.functional.mse_loss(input_vgg, target_vgg)
        return loss


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.vgg_loss = VGGLoss()
        self.adversarial_loss = nn.BCEWithLogitsLoss()

    def forward(self, fake_pred, real_pred, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = self.adversarial_loss(fake_pred - torch.mean(real_pred), torch.ones_like(fake_pred))
        # Pixel-wise Loss
        l1_loss = self.l1_loss(out_images, target_images)
        # VGG Loss
        # vgg_loss = self.vgg_loss(out_images, target_images)
        return l1_loss + 0.001 * adversarial_loss


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, real_pred, fake_pred):
        # Adversarial loss for real images (RaGAN)
        real_loss = self.criterion(real_pred - fake_pred.mean(0, keepdim=True), torch.ones_like(real_pred))
        real_loss = torch.mean(real_loss)

        # Adversarial loss for fake images (RaGAN)
        fake_loss = self.criterion(fake_pred - real_pred.mean(0, keepdim=True), torch.zeros_like(fake_pred))
        fake_loss = torch.mean(fake_loss)

        # Total discriminator loss
        d_loss = (real_loss + fake_loss) / 2
        return d_loss