import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

dataset, info = tfds.load('cycle_gan/horse2zebra', with_info=True, as_supervised=True)
train_horses, train_zebras = dataset['trainA'], dataset['trainB']
test_horses, test_zebras = dataset['testA'], dataset['testB']
print(info)

AUTOTUNE = tf.data.experimental.AUTOTUNE
BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3
EPOCHS = 40


def random_crop(image):
    cropped_image = tf.image.random_crop(
        image, size=[IMG_HEIGHT, IMG_WIDTH, 3])
    return cropped_image


# 将图像归一化到区间 [-1, 1] 内。
def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image


def random_jitter(image):
    # 调整大小为 286 x 286 x 3
    image = tf.image.resize(image, [286, 286],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # 随机裁剪到 256 x 256 x 3
    image = random_crop(image)

    # 随机镜像
    image = tf.image.random_flip_left_right(image)

    return image


def preprocess_image_train(image, label):
    image = random_jitter(image)
    image = normalize(image)
    return image


def preprocess_image_test(image, label):
    image = normalize(image)
    return image


def create_generator():
    return pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')


def create_discriminator():
    return pix2pix.discriminator(norm_type='instancenorm', target=False)


train_horses = train_horses.map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(1)

train_zebras = train_zebras.map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(1)

test_horses = test_horses.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(1)

test_zebras = test_zebras.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(1)


class CycleGan(object):
    def __init__(self, enable_function, epochs):
        """

        :param enable_function:
        :param epochs:
        """

        self.enable_function = enable_function
        self.epochs = epochs

        # g_g  :  X ----> Y  生成器 G 学习将图片 X 转换为 Y
        # g_f  :  Y ----> X  生成器 F 学习将图片 Y 转换为 X

        # d_X  :  判别器 D_X 学习区分图片 X 与生成的图片 X (F(Y))。
        # d_Y  :  判别器 D_Y 学习区分图片 Y 与生成的图片 Y (G(X))。

        self.generator_g = create_generator()
        self.generator_f = create_generator()
        self.discriminator_x = create_discriminator()
        self.discriminator_y = create_discriminator()

        self.LAMBDA = 10
        self.binary_loss_obj = tf.keras.losses.binary_crossentropy
        self.mse_loss_obj = tf.keras.losses.mse

        self.generator_opt_obj = tf.keras.optimizers.Adam()
        self.discriminator_opt_obj = tf.keras.optimizers.Adam()

    def discriminator_loss(self, real_img, gen_img):
        real_loss = self.binary_loss_obj(tf.ones_like(real_img), real_img)
        fake_loss = self.binary_loss_obj(tf.zeros_like(gen_img), gen_img)
        return real_loss + fake_loss

    def generator_loss(self, gen_img):
        return self.binary_loss_obj(tf.ones_like(gen_img), gen_img)

    def cycled_loss(self, real_img, cycled_img):
        # return self.LAMBDA * self.mse_loss_obj(real_img, cycled_img)
        return self.LAMBDA * tf.reduce_mean(tf.abs(real_img - cycled_img))

    def identity_loss(self, real_image, same_image):
        # return self.LAMBDA * 0.5 * self.mse_loss_obj(real_image, same_image)
        return self.LAMBDA * 0.5 * tf.reduce_mean(tf.abs(real_image - same_image))

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape(persistent=True) as tape:
            gen_x = self.generator_f(y)
            gen_y = self.generator_g(x)

            cycled_x = self.generator_f(gen_y)
            cycled_y = self.generator_g(gen_x)

            same_x = self.generator_f(x)
            same_y = self.generator_g(y)
            print(same_x)
            print(same_y)

            disc_x = self.discriminator_x(x)
            disc_y = self.discriminator_y(y)

            disc_gen_x = self.discriminator_x(gen_x)
            disc_gen_y = self.discriminator_y(gen_y)

            loss_disc_x = self.discriminator_loss(disc_x, disc_gen_x)
            loss_disc_y = self.discriminator_loss(disc_y, disc_gen_y)

            loss_generator_g = self.generator_loss(disc_gen_y)
            loss_generator_f = self.generator_loss(disc_gen_x)

            loss_total_cycle = self.cycled_loss(x, cycled_x) + self.cycled_loss(y, cycled_y)

            # 总生成器损失 = 对抗性损失 + 循环损失。
            print("----------------------------------\n")
            print( loss_generator_g)
            print(loss_total_cycle)
            loss_total_generator_g = loss_generator_g + loss_total_cycle + self.identity_loss(y, same_y)
            loss_total_generator_f = loss_generator_f + loss_total_cycle + self.identity_loss(x, same_x)

        grads_disc_x = tape.gradient(loss_disc_x, self.discriminator_x.trainable_variables)
        grads_disc_y = tape.gradient(loss_disc_y, self.discriminator_y.trainable_variables)
        grads_generator_g = tape.gradient(loss_total_generator_g, self.generator_g.trainable_variables)
        grads_generator_f = tape.gradient(loss_total_generator_f, self.generator_f.trainable_variables)

        self.discriminator_opt_obj.apply_gradients(zip(grads_disc_x, self.discriminator_x.trainable_variables))
        self.discriminator_opt_obj.apply_gradients(zip(grads_disc_y, self.discriminator_y.trainable_variables))
        self.generator_opt_obj.apply_gradients(zip(grads_generator_g, self.generator_g.trainable_variables))
        self.generator_opt_obj.apply_gradients(zip(grads_generator_f, self.generator_f.trainable_variables))

    def train(self):
        for epoch in range(self.epochs):
            for image_x, image_y in tf.data.Dataset.zip((train_horses, train_zebras)):
                print(image_x)
                self.train_step(image_x, image_y)


gan = CycleGan(True, EPOCHS)
gan.train()
