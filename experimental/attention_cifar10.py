import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

IMAGE_SHAPE = (32, 32, 3)
BATCH_SIZE = 32
EPOCHS = 5


def create_ds():
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    (ds_train, ds_test), info = tfds.load("cifar10", split=[tfds.Split.TRAIN, tfds.Split.TEST], with_info=True)
    ds_train = ds_train.shuffle(50000).map(lambda x: (tf.cast(x["image"], tf.float32) / 255., x["label"]),
                                           AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    ds_test = ds_test.map(lambda x: (tf.cast(x["image"], tf.float32) / 255., x["label"]), AUTOTUNE).batch(
        BATCH_SIZE).prefetch(AUTOTUNE)
    return ds_train, ds_test


def make_model():
    input_img = tf.keras.layers.Input(shape=IMAGE_SHAPE)

    attention_pred = tf.keras.layers.Conv2D(3, 1, 1, padding="same", activation=tf.keras.activations.softmax)(input_img)
    # attention_pred = tf.keras.activations.softmax(attention_conv)

    compose = tf.keras.layers.Multiply()([input_img, attention_pred])
    x = tf.keras.layers.Conv2D(32, 2, 2, padding="same")(compose)
    x = tf.keras.layers.Conv2D(16, 2, 2, padding="same")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(32, tf.keras.activations.relu)(x)
    x = tf.keras.layers.Dense(10, tf.keras.activations.softmax)(x)
    m = tf.keras.models.Model(input_img, x)
    m.summary()
    tf.keras.utils.plot_model(m, to_file="attention_cifar10.png", show_shapes=True)
    return m


def extract(model, layer_name, x):
    out = model.get_layer(layer_name).output
    print(out)
    extractor = tf.keras.models.Model(model.input, out)
    return extractor(x)


ds_train, ds_test = create_ds()
model = make_model()
for l in model.layers:
    print(l.name)
model.compile("adam", tf.keras.losses.sparse_categorical_crossentropy, ["acc"])
# model.fit(ds_train, validation_data=ds_test, epochs=EPOCHS)
test_data = next(iter(ds_test))[0]
attention_vector = extract(model, "conv2d", test_data)
# attention_vector = extract(model, "tf_op_layer_truediv", test_data)
print(attention_vector)

plt.subplot(1, 2, 1)
plt.imshow(test_data[0])

plt.subplot(1, 2, 2)
plt.imshow(attention_vector[0])
plt.show()
