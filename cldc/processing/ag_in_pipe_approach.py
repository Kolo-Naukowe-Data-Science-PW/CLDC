from functools import partial
import albumentations as A
import tensorflow as tf


class DatasetReader:
    def __init__(self, img_size=(512, 512)):

        self.IMAGE_SIZE = img_size

    def _decode_image(self, image):
        image_raw = tf.image.decode_jpeg(image, channels=3)
        image_raw = tf.cast(image_raw, tf.float32)

        # array with float values 0..1
        image_raw = tf.reshape(image_raw, [*self.IMAGE_SIZE, 3]) / 255.0

        return image_raw

    def _read_tfrecord(self, example):
        # image feature keys inside tfrecord files
        image_features = {
            'target': tf.io.FixedLenFeature([], tf.int64),
            'image': tf.io.FixedLenFeature([], tf.string),
        }

        example = tf.io.parse_single_example(example, image_features)

        # image in Tensor format
        image = self._decode_image(example["image"])

        # class of decoded image
        label = tf.cast(example["target"], tf.int32)

        return image, label

    def _load_dataset(self, filenames, num_parallel_calls):
        ignore_order = tf.data.Options()

        # disable order, increase speed
        ignore_order.experimental_deterministic = False

        # automatically interleaves reads from multiple files
        dataset = tf.data.TFRecordDataset(filenames)

        # uses data as soon as it streams in, rather than in its original order
        dataset = dataset.with_options(ignore_order)

        # returns a dataset of (image, label) pairs
        return dataset.map(self._read_tfrecord, num_parallel_calls=num_parallel_calls)

    def get_dataset(self, filenames, shuffle_size=False, batch_size=False, num_parallel_calls=tf.data.experimental.AUTOTUNE):
        # filenames in format ['file1.tfrec', 'file2.tfrec'...]
        dataset = self._load_dataset(filenames, num_parallel_calls)
        if type(shuffle_size) is int:
            dataset = dataset.shuffle(shuffle_size)
        if type(batch_size) is int:
            dataset = dataset.batch(batch_size)
        return dataset

    def get_splitted_dataset(self, filenames, dataset_size=21397, ratios=(0.7, 0.2, 0.1)):
        """
        Loads and splits dataset at given ratio.
        :return: tuple (train_dataset, validation_dataset, test_dataset)
        """
        full_dataset = self.get_dataset(filenames)
        train_size = int(ratios[0] * dataset_size)
        val_size = int(ratios[1] * dataset_size)
        test_size = int(ratios[2] * dataset_size)

        train_dataset = full_dataset.take(train_size)
        test_dataset = full_dataset.skip(train_size)
        val_dataset = test_dataset.skip(test_size)
        test_dataset = test_dataset.take(test_size)

        return train_dataset, val_dataset, test_dataset


class Augmentator:
    def __init__(self, aug_composition='default'):
        if aug_composition == 'default':
            self.aug_composition = A.Compose([
                A.Blur(blur_limit=4, p=0.2),
                A.Rotate(always_apply=True),
                A.RandomBrightnessContrast(brightness_limit=(-0.09, 0.26),
                                           contrast_limit=(-0.24, 0.13), p=0.5),
                A.HorizontalFlip(),
            ])
        else:
            self.aug_composition = aug_composition

    def augment(self, dataset, image_shape=(512, 512)):
        """
        Carry out augmentation for given tf.data.Dataset.
        :param dataset: tf.data.Dataset with singular examples as tuples (image, label) with dtypes (tf.float32, tf.int32)
        :param image_shape: Tuple with image size (height, width)
        :return: tf.data.Dataset with singular examples as tuples (image, label) with dtypes (tf.float32, tf.int32) with augumented examples
        """
        AUTOTUNE = tf.data.experimental.AUTOTUNE

        def aug_fn(image, img_size):
            data = {"image": image}
            aug_data = self.aug_composition(**data)
            aug_img = aug_data["image"]
            aug_img = tf.cast(aug_img, tf.float32)
            aug_img = tf.image.resize(aug_img, size=img_size)
            return aug_img

        # wrapping augmentation function to numpy mapping function
        def process_data(image, label, img_size):
            aug_img = tf.numpy_function(func=aug_fn, inp=[image, img_size], Tout=tf.float32)
            return aug_img, label

        # mapping function for restoring shape
        def set_shapes(img, label, img_shape=(*image_shape, 3)):
            img.set_shape(img_shape)
            label.set_shape([])
            return img, label

        aug_ds = dataset.map(partial(process_data, img_size=image_shape), num_parallel_calls=AUTOTUNE)
        aug_ds = aug_ds.map(partial(set_shapes, img_shape=(*image_shape, 3)), num_parallel_calls=AUTOTUNE)

        return aug_ds


def one_hot_encode(image, label, class_n):
    """
    Function performing one hot encoding in tf.data.Dataset pipeline.
    usage: train_ds = train_ds.map(partial(one_hot_encode, class_n=5), num_parallel_calls=AUTOTUNE)
    """
    label = tf.one_hot(label, class_n)
    return image, label
