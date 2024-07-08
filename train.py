import os
from math import log10

import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import pytorch_ssim
from dataloader import TrainDatasetFromFolder, ValDatasetFromFolder
from loss import GeneratorLoss, DiscriminatorLoss
from model import Generator, Discriminator
from datetime import datetime
from torch.nn import BCEWithLogitsLoss


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

if __name__ == '__main__':
    CROP_SIZE = 256
    NUM_EPOCHS = 50
    BATCH_SIZE = 8  # 减少批量大小
    NUM_WORKERS = 4
    COLOR_MODE = 'L'


    # 创建数据加载器,使用pin_memory=True
    train_set = TrainDatasetFromFolder('dataset_restore', crop_size=CROP_SIZE, color_mode=COLOR_MODE)
    train_loader = DataLoader(dataset=train_set, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_set = ValDatasetFromFolder('val_images', crop_size=CROP_SIZE, color_mode=COLOR_MODE)
    val_loader = DataLoader(dataset=val_set, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # 初始化生成器和判别器
    netG = Generator(color_mode=COLOR_MODE)
    netD = Discriminator(color_mode=COLOR_MODE)

    # 检查CUDA是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG.to(device)
    netD.to(device)

    # 定义损失函数和优化器
    generator_criterion = GeneratorLoss().to(device)
    discriminator_criterion = DiscriminatorLoss().to(device)
    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters(), lr=0.001)

    # 创建TensorBoard写入器
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join('runs', current_time)
    print(f"TensorBoard logs will be saved in {log_dir}")
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

        netG.train()
        netD.train()
        for target, data in train_bar:
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            # 将输入和目标数据加载到GPU上
            real_img = target.to(device, non_blocking=True)
            z = data.to(device, non_blocking=True)
            fake_img = netG(z)

            # 更新判别器
            optimizerD.zero_grad()
            real_pred = netD(real_img)
            fake_pred = netD(fake_img.detach())
            d_loss = discriminator_criterion(real_pred, fake_pred)
            d_loss.backward()
            optimizerD.step()

            # 更新生成器
            optimizerG.zero_grad()
            real_pred = netD(real_img)
            fake_pred = netD(fake_img)
            g_loss = generator_criterion(fake_pred, real_pred, fake_img, real_img)
            g_loss.backward()
            optimizerG.step()

            # 记录当前批次的损失和分数
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_pred.mean().item() * batch_size
            running_results['g_score'] += fake_pred.mean().item() * batch_size

            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))

        # 在每个epoch结束时,使用验证集图像生成图像并保存到TensorBoard
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            val_images = []
            val_targets = []
            for val_target, val_data in val_bar:
                val_images.append(val_data.to(device, non_blocking=True))
                val_targets.append(val_target.to(device, non_blocking=True))
            val_images = torch.cat(val_images, dim=0)
            val_targets = torch.cat(val_targets, dim=0)
            fake_val_images = netG(val_images).detach().cpu()
            val_images = val_images.cpu()
            img_grid_fake = utils.make_grid(fake_val_images, normalize=True)
            img_grid_real = utils.make_grid(val_targets.cpu(), normalize=True)
            writer.add_image(f'Fake Images', img_grid_fake, global_step=epoch)
            writer.add_image(f'Real Images', img_grid_real, global_step=epoch)


        # 保存模型参数
        torch.save(netG.state_dict(), f'checkpoints/netG_epoch_{epoch}.pth')
        torch.save(netD.state_dict(), f'checkpoints/netD_epoch_{epoch}.pth')

        # 将每个epoch的损失和分数写入TensorBoard
        writer.add_scalar('Loss/Discriminator', running_results['d_loss'] / running_results['batch_sizes'], epoch)
        writer.add_scalar('Loss/Generator', running_results['g_loss'] / running_results['batch_sizes'], epoch)
        writer.add_scalar('Score/Real', running_results['d_score'] / running_results['batch_sizes'], epoch)
        writer.add_scalar('Score/Fake', running_results['g_score'] / running_results['batch_sizes'], epoch)

        # 清空GPU缓存
        torch.cuda.empty_cache()

    writer.close()