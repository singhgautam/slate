from torch.utils.data import Dataset
import h5py
import torch


class Shapes3D(Dataset):
    def __init__(self, root, phase):
        assert phase in ['train', 'val', 'test']
        with h5py.File(root, 'r') as f:
            if phase == 'train':
                self.imgs = f['images'][:400000]
            elif phase == 'val':
                self.imgs = f['images'][400001:430000]
            elif phase == 'test':
                self.imgs = f['images'][430001:460000]
            else:
                raise NotImplementedError

    def __getitem__(self, index):
        img = self.imgs[index]
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = img.float() / 255.

        return img

    def __len__(self):
        return len(self.imgs)
