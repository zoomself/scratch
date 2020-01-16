import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_data(n, input_dim, attention_column=1):
    x = np.random.standard_normal(size=(n, input_dim))
    y = np.random.randint(low=0, high=2, size=(n, 1))

    x[:, attention_column] = y[:, 0]
    print(x)
    print(y)
    return x, y


def get_attention(model, name, x):
    m = tf.keras.models.Model(model.input, model.get_layer(name).output)
    return m(x)


# INPUT_DIM = 32
# N = 10000
#
# input_img = tf.keras.Input(shape=(INPUT_DIM,))
#
# attention_probs = tf.keras.layers.Dense(INPUT_DIM, activation=tf.keras.activations.softmax, name="attention")(input_img)
# attention_mul = tf.keras.layers.Multiply()([input_img, attention_probs])
# print("attention_mul=", attention_mul)
# attention_mul = tf.keras.layers.Dense(INPUT_DIM * 2)(attention_mul)
# output = tf.keras.layers.Dense(1, activation='sigmoid')(attention_mul)
# model = tf.keras.models.Model(input_img, output)
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.summary()
#
# tf.keras.utils.plot_model(model, show_shapes=True)
#
# inputs_1, outputs = get_data(N, INPUT_DIM)
# print(inputs_1)
# print(outputs)
#
# model.fit([inputs_1], outputs, epochs=20, batch_size=32, validation_split=0.5)
#
# testing_inputs_1, testing_outputs = get_data(1, INPUT_DIM)
#
# # Attention vector corresponds to the second matrix.
# # The first one is the Inputs output.
# attention_vector = get_attention(model, "attention", testing_inputs_1)[0]
# # attention_vector = get_activations(model, testing_inputs_1,
# #                                    print_shape_only=True,
# #                                    layer_name='attention_vec')[0].flatten()
# print('attention =', attention_vector)
#
# # plot part.
#
#
# pd.DataFrame(attention_vector, columns=['attention (%)']).plot(kind='bar', title='Attention Mechanism as '
#                                                                                  'a function of input'
#                                                                                  ' dimensions.')
# plt.show()

LATENT_DIM = 1


def create_model():
    input_img = tf.keras.layers.Input(shape=(28, 28, 1))

    temp_img = tf.keras.layers.Input(shape=(LATENT_DIM,))

    attention_prob = tf.keras.layers.Dense(28 * 28, activation=tf.keras.activations.softmax)(temp_img)
    attention_reshape = tf.keras.layers.Reshape(target_shape=(28, 28, 1), name="attention")(attention_prob)

    x = tf.keras.layers.multiply([input_img, attention_reshape])

    x = tf.keras.layers.Conv2D(16, 3, 2, padding="same")(x)
    x = tf.keras.layers.Conv2D(8, 3, 2, padding="same")(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(10, tf.keras.activations.softmax)(x)
    m = tf.keras.models.Model([input_img, temp_img], x)
    m.summary()
    tf.keras.utils.plot_model(m, show_shapes=True)

    return m


(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
train_x = train_x / 255.
train_x = train_x[..., tf.newaxis]
test_x = test_x / 255.
test_x = test_x[..., tf.newaxis]

print(train_y.shape)
print(test_y.shape)

model = create_model()
model.compile("adam", tf.keras.losses.sparse_categorical_crossentropy, ["acc"])
model.fit((train_x, train_y), train_y, validation_freq=0.2, batch_size=32, epochs=5)

attention_vector = get_attention(model, "attention", (test_x, test_y[..., tf.newaxis]))[:4]
attention_vector = tf.squeeze(attention_vector, axis=-1)
# print(attention_vector)
plt.subplot(2, 4, 1)
plt.imshow(tf.squeeze(test_x[0], -1))
plt.subplot(2, 4, 2)
plt.imshow(attention_vector[0])

plt.subplot(2, 4, 3)
plt.imshow(tf.squeeze(test_x[1], -1))
plt.subplot(2, 4, 4)
plt.imshow(attention_vector[1])

plt.subplot(2, 4, 5)
plt.imshow(tf.squeeze(test_x[2], -1))
plt.subplot(2, 4, 6)
plt.imshow(attention_vector[2])

plt.subplot(2, 4, 7)
plt.imshow(tf.squeeze(test_x[3], -1))
plt.subplot(2, 4, 8)
plt.imshow(attention_vector[3])
plt.show()
