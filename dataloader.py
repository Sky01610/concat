from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def crop_image(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])


def convert_to_gray(image):
    return image.convert('L')

def convert_to_LA(image):
    return image.convert('LA')

def convert_to_color(image):
    return image.convert('RGB')


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, color_mode='RGB'):
        super(TrainDatasetFromFolder, self).__init__()
        self.color_mode = color_mode
        clean_dataset_dir = join(dataset_dir, 'clean')
        noisy_dataset_dir = join(dataset_dir, 'noisy')
        self.clean_img_filenames = [join(clean_dataset_dir, x) for x in listdir(clean_dataset_dir) if is_image_file(x)]
        self.noisy_img_filenames = [join(noisy_dataset_dir, x) for x in listdir(noisy_dataset_dir) if is_image_file(x)]
        self.crop_transform = crop_image(crop_size)

    def __getitem__(self, index):
        clean_img = Image.open(self.clean_img_filenames[index])
        noisy_img = Image.open(self.noisy_img_filenames[index])

        if self.color_mode == 'L':
            clean_img = convert_to_gray(clean_img)
            noisy_img = convert_to_gray(noisy_img)
        if self.color_mode == 'LA':
            clean_img = convert_to_LA(clean_img)
            noisy_img = convert_to_LA(noisy_img)
        elif self.color_mode == 'RGB':
            clean_img = convert_to_color(clean_img)
            noisy_img = convert_to_color(noisy_img)

        clean_img = self.crop_transform(clean_img)
        noisy_img = self.crop_transform(noisy_img)

        return clean_img, noisy_img

    def __len__(self):
        return len(self.clean_img_filenames)

class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, color_mode='RGB'):
        super(ValDatasetFromFolder, self).__init__()
        self.color_mode = color_mode
        clean_dataset_dir = join(dataset_dir, 'clean')
        noisy_dataset_dir = join(dataset_dir, 'noisy')
        self.clean_img_filenames = [join(clean_dataset_dir, x) for x in listdir(clean_dataset_dir) if is_image_file(x)]
        self.noisy_img_filenames = [join(noisy_dataset_dir, x) for x in listdir(noisy_dataset_dir) if is_image_file(x)]
        self.crop_transform = crop_image(crop_size)

    def __getitem__(self, index):
        clean_img = Image.open(self.clean_img_filenames[index])
        noisy_img = Image.open(self.noisy_img_filenames[index])

        if self.color_mode == 'L':
            clean_img = convert_to_gray(clean_img)
            noisy_img = convert_to_gray(noisy_img)
        if self.color_mode == 'LA':
            clean_img = convert_to_LA(clean_img)
            noisy_img = convert_to_LA(noisy_img)
        elif self.color_mode == 'RGB':
            clean_img = convert_to_color(clean_img)
            noisy_img = convert_to_color(noisy_img)

        clean_img = self.crop_transform(clean_img)
        noisy_img = self.crop_transform(noisy_img)

        return clean_img, noisy_img

    def __len__(self):
        return len(self.clean_img_filenames)