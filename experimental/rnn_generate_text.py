import tensorflow as tf
import numpy as np

path_to_file = tf.keras.utils.get_file('shakespeare.txt',
                                       'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text = open(path_to_file, encoding="utf-8").read()

text_set = sorted(set(text))
vocab_size = len(text_set)
char2idx = {u: i for i, u in enumerate(text_set)}
idx2char = np.array(text_set)
text_as_int = [char2idx[t] for t in text]

print(text_set)
print(char2idx)
print(idx2char)
print(text_as_int)

PER_SEQ_LENGTH = 101
AUTOTUNE = tf.data.experimental.AUTOTUNE
# 嵌入的维度
embedding_dim = 256
# RNN 的单元数量
rnn_units = 1024

BATCH_SIZE = 64

ds = tf.data.Dataset.from_tensor_slices(text_as_int) \
    .batch(PER_SEQ_LENGTH, True).map(lambda x: (x[:100], x[1:]), AUTOTUNE).batch(BATCH_SIZE, True) \
    .prefetch(AUTOTUNE)

print(next(iter(ds)))

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=(BATCH_SIZE, PER_SEQ_LENGTH - 1),
                              input_shape=(PER_SEQ_LENGTH - 1,)),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
])
model.summary()
(data_x, data_y) = next(iter(ds))
pred_y = model(data_x)
print(pred_y.shape)
sample_y = tf.random.categorical(pred_y[0], 1)
sample_y = tf.squeeze(sample_y, -1)
print(sample_y)
print("".join(idx2char[sample_y]))


def loss(y_true, y_pred):
    return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, True)


model.compile("adam", loss)
model.fit(ds, epochs=10)
