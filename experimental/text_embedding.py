import tensorflow as tf
import tensorflow_datasets as tfds

(ds_train, ds_test), info = tfds.load(
    'imdb_reviews/subwords8k',
    split=(tfds.Split.TRAIN, tfds.Split.TEST),
    with_info=True, as_supervised=True)
print(info)
print(next(iter(ds_train)))
ds_train = ds_train.padded_batch(16, padded_shapes=([-1], ())).prefetch(tf.data.experimental.AUTOTUNE)
ds_test = ds_test.padded_batch(16, padded_shapes=([-1], ())).prefetch(tf.data.experimental.AUTOTUNE)

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(8187, 16, input_shape=(None,)),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(1, tf.keras.activations.sigmoid)
])
model.summary()
model.compile("adam", tf.keras.losses.binary_crossentropy, ["acc"])
model.fit(ds_train, epochs=10, validation_data=ds_test)
