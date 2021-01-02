import torch


class RegularDataset(torch.utils.data.Dataset):
    def __init__(self, iter, length):
        super().__init__()
        self.iter = iter
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return next(self.iter)


if __name__ == "__main__":
    steps_per_epoch = 1
    train_dataset = None
    train_loader = torch.utils.data.DataLoader(
        RegularDataset(
            train_dataset.as_numpy_iterator(),
            length=steps_per_epoch,
        ),
        num_workers=0,
        batch_size=None,
    )
