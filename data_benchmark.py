import os
import time
from math import ceil

import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm

from pytorch_imp.data import get_dataloader
from tensorflow_imp.data import get_dataset


def main():
    data_dir = "gs://tf-vs-torch/test-data/images/pokemon_jpg"
    data_dir = "data/pokemon_jpg"
    image_paths = [
        os.path.join(data_dir, file) for file in tf.io.gfile.listdir(data_dir)
    ]

    image_paths = image_paths[:100]

    batch_size = 32
    num_workers = 4
    prefetch_factor = 2
    epochs = 10
    time_per_step = 0.01

    num_steps = ceil(len(image_paths) / batch_size)

    # tensorflow
    dataset = get_dataset(
        image_paths,
        load_num_parallel_calls=num_workers,
        batch_size=batch_size,
        shuffle=True,
        cache_dir=None,
        # cache_dir="train",
        repeat=False,
        prefetch_factor=prefetch_factor,
    )

    init_tf = time.time()
    for epoch in tqdm(range(epochs), total=epochs, desc="Epoch: "):
        for step, batch in tqdm(enumerate(dataset), total=num_steps, desc="Step: "):
            time.sleep(time_per_step)
    total_tf = time.time() - init_tf
    print(f"Total time tensorflow: {total_tf}")
    # exit()

    # pytorch
    dataset = get_dataloader(
        image_paths,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
        prefetch_factor=prefetch_factor,
    )

    init_torch = time.time()
    for epoch in tqdm(range(epochs), total=epochs, desc="Epoch: "):
        for step, batch in tqdm(enumerate(dataset), total=num_steps, desc="Step: "):
            time.sleep(time_per_step)
    total_torch = time.time() - init_torch
    print(f"Total time pytorch: {total_torch}")


if __name__ == "__main__":
    main()
