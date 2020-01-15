import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_data(n, input_dim, attention_column=2):
    x = np.random.standard_normal(size=(n, input_dim))
    y = np.random.randint(low=0, high=2, size=(n, 1))
    x[:, attention_column] = y[:, 0]
    return x, y


def get_attention(model, name, x):
    m = tf.keras.models.Model(model.input, model.get_layer(name).output)
    return m(x)


INPUT_DIM = 32
N = 10000

input_img = tf.keras.Input(shape=(INPUT_DIM,))

attention_probs = tf.keras.layers.Dense(INPUT_DIM, activation=tf.keras.activations.softmax, name="attention")(input_img)
# attention_mul = tf.keras.layers.concatenate([input_img, attention_probs], axis=-2)
# attention_mul = tf.multiply(input_img, attention_probs)
attention_mul = tf.keras.layers.Multiply()([input_img, attention_probs])
print("attention_mul=", attention_mul)
attention_mul = tf.keras.layers.Dense(INPUT_DIM * 2)(attention_mul)
output = tf.keras.layers.Dense(1, activation='sigmoid')(attention_mul)
model = tf.keras.models.Model(input_img, output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

tf.keras.utils.plot_model(model, show_shapes=True)

inputs_1, outputs = get_data(N, INPUT_DIM)
print(inputs_1)
print(outputs)

model.fit([inputs_1], outputs, epochs=20, batch_size=32, validation_split=0.5)

testing_inputs_1, testing_outputs = get_data(1, INPUT_DIM)

# Attention vector corresponds to the second matrix.
# The first one is the Inputs output.
attention_vector = get_attention(model, "attention", testing_inputs_1)[0]
# attention_vector = get_activations(model, testing_inputs_1,
#                                    print_shape_only=True,
#                                    layer_name='attention_vec')[0].flatten()
print('attention =', attention_vector)

# plot part.


pd.DataFrame(attention_vector, columns=['attention (%)']).plot(kind='bar', title='Attention Mechanism as '
                                                                                 'a function of input'
                                                                                 ' dimensions.')
plt.show()
