import tensorflow as tf


class DatasetReader:
    def __init__(self,  batch_size=16):
        self.BATCH_SIZE = batch_size

        # default size in cassava dataset
        self.IMAGE_SIZE = [512, 512]

        self.dataset = None

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

    def _load_dataset(self, filenames):
        ignore_order = tf.data.Options()

        # disable order, increase speed
        ignore_order.experimental_deterministic = False

        # automatically interleaves reads from multiple files
        dataset = tf.data.TFRecordDataset(filenames)

        # uses data as soon as it streams in, rather than in its original order
        dataset = dataset.with_options(ignore_order)

        # returns a dataset of (image, label) pairs 
        return dataset.map(self._read_tfrecord)

    def get_dataset(self, filenames):
        AUTOTUNE = tf.data.experimental.AUTOTUNE

        dataset = self._load_dataset(filenames)
        dataset = dataset.shuffle(2048)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        dataset = dataset.batch(self.BATCH_SIZE)
        return dataset
