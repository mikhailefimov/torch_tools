import numpy as np
import torch
from torch.utils.data import Dataset


class RandomImageDataset(Dataset):
    def __init__(self, width=224, height=224, count=5, nclasses=2, detection=True):
        self.width = width
        self.height = height
        self.count = count
        self.detection = detection
        self.nclasses = nclasses

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        np.random.seed(index + 42)
        back_color = np.random.randint(0, 128)
        count = np.random.randint(1, 5)
        img = np.full((self.height, self.width, 3), back_color, dtype=np.uint8)
        boxes = []
        labels = []
        for i in range(count):
            front_color = back_color + np.random.randint(64, 128)
            w = np.random.randint(self.width // 8, (7 * self.width) // 8)
            h = np.random.randint(self.height // 8, (7 * self.height) // 8)
            x = np.random.randint(0, self.width - w)
            y = np.random.randint(0, self.height - h)
            img[y:y + h, x:x + w, :] = np.array((front_color, front_color - 10, front_color - 20))
            if self.detection:
                boxes.append((x, y, x + w, y + h))
                labels.append(np.random.randint(0, self.nclasses))
        if self.detection:
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.long)
            }
        else:
            target = np.random.randint(0, self.nclasses)
        img = torch.from_numpy((img / 256).astype(np.float32)).permute(2, 0, 1)
        return img, target
