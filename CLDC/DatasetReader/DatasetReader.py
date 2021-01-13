import tensorflow as tf
from functools import partial


class DatasetReader:
    def __init__(self):
        # default size in cassava dataset
        self.IMAGE_SIZE = [512, 512]

        self.dataset = None

    def _decode_image(self, image):
        image_raw = tf.image.decode_jpeg(image, channels=3)
        image_raw = tf.cast(image_raw, tf.float32)

        # array with float values 0..1
        image_raw = tf.reshape(image_raw, [*self.IMAGE_SIZE, 3]) / 255.0

        return image_raw

    def _read_tfrecord(self, example, labeled):
        # image feature keys inside tfrecord files
        if labeled:
            image_features = {
            'target': tf.io.FixedLenFeature([], tf.int64),
            'image': tf.io.FixedLenFeature([], tf.string),
            }
        else:
             image_features = {
            'image': tf.io.FixedLenFeature([], tf.string),
            }

        example = tf.io.parse_single_example(example, image_features)

        # image in Tensor format
        image = self._decode_image(example["image"])
        
        if labeled:
            # class of decoded image
            label = tf.cast(example["target"], tf.int32)
            return image, label

        return image

    def _load_dataset(self, filenames, labeled):
        ignore_order = tf.data.Options()

        # disable order, increase speed
        ignore_order.experimental_deterministic = False

        # automatically interleaves reads from multiple files
        dataset = tf.data.TFRecordDataset(filenames)

        # uses data as soon as it streams in, rather than in its original order
        dataset = dataset.with_options(ignore_order)

        # returns a dataset of (image, label) pairs if labeled=True. If False returns only image
        return dataset.map(partial(self._read_tfrecord, labeled=labeled))

    def get_dataset(self, filenames, batch_size=16, labeled=True):
        """Return dataset from given file

        Arguments:
        filenames -- list of files to read
        batch_size -- None or int. If None, dataset won't be batched.
        labeled -- bool, default True and returns dataset as pairs of (image, label). If False returns only image

        Returns:
        dataset
        """

        AUTOTUNE = tf.data.experimental.AUTOTUNE

        dataset = self._load_dataset(filenames, labeled)
        dataset = dataset.shuffle(2048)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)

        if labeled:
            dataset = dataset.batch(batch_size)

        return dataset
