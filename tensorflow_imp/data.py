import tensorflow as tf

import os
import shutil
import typing as tp


def read_image(image_path: str) -> tf.Tensor:
    image = tf.io.read_file(image_path)
    image = tf.io.decode_image(image, expand_animations=False)
    return image


def get_dataset(
    image_paths: str,
    load_num_parallel_calls: int = 4,
    batch_size: int = 32,
    shuffle: bool = True,
    cache_dir: tp.Optional[str] = None,
    repeat: bool = True,
    prefetch_factor: tp.Optional[str] = 4,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)

    def load_row(row):
        image = read_image(row)
        return image

    dataset = dataset.map(load_row, num_parallel_calls=load_num_parallel_calls)

    if shuffle:
        dataset = dataset.shuffle(100)

    # if cache dir assume caching
    if cache_dir is not None:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir)
        dataset = dataset.cache(os.path.join(cache_dir, "cache"))

    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size, drop_remainder=False)

    if prefetch_factor:
        dataset = dataset.prefetch(prefetch_factor)

    return dataset


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    data_dir = "gs://tf-vs-torch/test-data/images/pokemon_jpg"
    # data_dir = "data/pokemon_jpg"

    image_paths = [
        os.path.join(data_dir, file) for file in tf.io.gfile.listdir(data_dir)
    ]
    # print(image_paths)
    dataset = get_dataset(
        image_paths[:100],
        load_num_parallel_calls=4,
        batch_size=32,
        shuffle=True,
        cache_dir=None,
        # cache_dir="train",
        repeat=False,
        prefetch_factor=2,
    )
    for batch in dataset:
        print(batch.shape)
        plt.imshow(batch[0].numpy())
        plt.show()

    for batch in dataset:
        print(batch.shape)
