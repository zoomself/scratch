import tensorflow as tf

# 5000张二次元头像生成
SMALL_DATA_FILE_PATTERN = "E:/dataset/anime/faces_test/*.jpg"
SMALL_RECORD_FILE = "./data/anime_5000.tfrecord"
SMALL_RECORD_FILE_BY_EXAMPLE = "./data/anime_5000_by_example.tfrecord"

# 51222张二次元头像生成
BIG_DATA_FILE_PATTERN = "E:/dataset/anime/faces/*.jpg"
BIG_RECORD_FILE = "./data/anime_51222.tfrecord"
BIG_RECORD_FILE_BY_EXAMPLE = "./data/anime_51222_by_example.tfrecord"


class Creator(object):
    def __init__(self, record_by_dataset_saved_file=None, record_by_example_saved_file=None):
        if record_by_dataset_saved_file is None and record_by_example_saved_file is None:
            raise FileNotFoundError
        self.record_by_dataset_saved_file = record_by_dataset_saved_file
        self.record_by_example_saved_file = record_by_example_saved_file

    def _create_by_dataset(self, data_file_pattern):
        ds_file = tf.data.Dataset.list_files(data_file_pattern).map(tf.io.read_file)
        tf.data.experimental.TFRecordWriter(self.record_by_dataset_saved_file).write(ds_file)

    def _create_by_example(self, data_file_pattern):
        file_paths = tf.io.gfile.glob(data_file_pattern)
        with tf.io.TFRecordWriter(self.record_by_example_saved_file) as writer:
            for path in file_paths:
                img = tf.io.read_file(path)
                feature = {
                    "img": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.numpy()]))
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

    def create(self, data_file_pattern):
        if data_file_pattern is None:
            raise FileNotFoundError
        if self.record_by_dataset_saved_file is not None:
            self._create_by_dataset(data_file_pattern)
        if self.record_by_example_saved_file is not None:
            self._create_by_example(data_file_pattern)

    def read(self):
        if self.record_by_dataset_saved_file is not None:
            return tf.data.TFRecordDataset(self.record_by_dataset_saved_file).map(
                lambda x: (tf.cast(tf.image.decode_jpeg(x), tf.float32)) / 255.,
                tf.data.experimental.AUTOTUNE)  # [-1,1]
        elif self.record_by_example_saved_file is not None:
            feature_description = {
                "img": tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            }
            return tf.data.TFRecordDataset(self.record_by_example_saved_file).map(
                lambda x: tf.io.parse_single_example(x, feature_description), tf.data.experimental.AUTOTUNE).map(
                lambda x: (tf.cast(tf.image.decode_jpeg(x["img"]), tf.float32) - 127.5) / 127.5,
                tf.data.experimental.AUTOTUNE)


def main():
    creator = Creator(record_by_dataset_saved_file=SMALL_RECORD_FILE,
                      record_by_example_saved_file=SMALL_RECORD_FILE_BY_EXAMPLE)
    creator.create(SMALL_DATA_FILE_PATTERN)

    # creator1 = Creator(record_by_dataset_saved_file=BIG_RECORD_FILE,
    #                    record_by_example_saved_file=BIG_RECORD_FILE_BY_EXAMPLE)
    #
    # creator1.create(BIG_DATA_FILE_PATTERN)


if __name__ == '__main__':
    main()
