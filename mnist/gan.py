import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app
from absl import flags

FLAGS = flags.FLAGS


def map_ds(x):
    x = x["image"]
    x = tf.cast(x, tf.float32) / 255.
    return x


def create_ds(batch_size):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    (ds_train, ds_test), info = tfds.load("mnist", with_info=True, split=[tfds.Split.TRAIN, tfds.Split.TEST])
    ds_train = ds_train.shuffle(60000).map(map_ds, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(
        AUTOTUNE)
    ds_test = ds_test.shuffle(10000).map(map_ds, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(
        AUTOTUNE)
    print(info)
    print(next(iter(ds_train)))
    return ds_train, ds_test


def create_generator():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(128 * 7 * 7, activation="relu"))
    model.add(tf.keras.layers.Reshape((7, 7, 128)))
    model.add(tf.keras.layers.UpSampling2D())
    model.add(tf.keras.layers.Conv2D(128, kernel_size=3, padding="same", activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(tf.keras.layers.UpSampling2D())
    model.add(tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(tf.keras.layers.Conv2D(1, kernel_size=3, padding="same", activation=tf.keras.activations.tanh))
    return model


def create_discriminator():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=3, strides=2,  padding="same"))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model


class Gan(object):
    def __init__(self, epochs, latent_dim, batch_size, check_point_dir, save_interval=100):
        self.epochs = epochs
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.save_interval = save_interval
        self.check_point_dir = check_point_dir
        if not tf.io.gfile.exists(check_point_dir):
            tf.io.gfile.makedirs(check_point_dir)

        self.generator = create_generator()
        self.discriminator = create_discriminator()

        self.loss_obj = tf.keras.losses.BinaryCrossentropy()
        self.optimizers_generator_obj = tf.keras.optimizers.Adam()
        self.optimizers_discriminator_obj = tf.keras.optimizers.Adam()

        self.metrics_acc_obj = tf.keras.metrics.BinaryAccuracy()

        self.global_step = tf.Variable(0, dtype=tf.int32)

        self.check_point = tf.train.Checkpoint(
            global_step=self.global_step,
            generator=self.generator,
            discriminator=self.discriminator,
            optimizers_generator_obj=self.optimizers_generator_obj,
            optimizers_discriminator_obj=self.optimizers_discriminator_obj,
        )
        self.check_point_manager = tf.train.CheckpointManager(self.check_point, directory=check_point_dir,
                                                              max_to_keep=3)
        if self.check_point_manager.latest_checkpoint:
            print("restore from check_point: {}".format(self.check_point_manager.latest_checkpoint))
            self.check_point.restore(self.check_point_manager.latest_checkpoint)
        else:
            print("Initializing from scratch...")

    def discriminator_loss(self, disc_gen_out, disc_real_out):
        l1 = self.loss_obj(tf.zeros_like(disc_gen_out), disc_gen_out)
        l2 = self.loss_obj(tf.ones_like(disc_real_out), disc_real_out)
        return l1 + l2

    def generator_loss(self, disc_gen_out):
        return self.loss_obj(tf.ones_like(disc_gen_out), disc_gen_out)

    @tf.function
    def train_step(self, real_img):
        with tf.GradientTape(persistent=True) as tape:
            batch = tf.shape(real_img)[0]
            noise = tf.random.normal(shape=(batch, self.latent_dim))
            gen_img = self.generator(noise)
            disc_real_out = self.discriminator(real_img)
            disc_gen_out = self.discriminator(gen_img)

            d_loss = self.discriminator_loss(disc_gen_out, disc_real_out)
            g_loss = self.generator_loss(disc_gen_out)

        self.metrics_acc_obj(tf.ones_like(disc_gen_out), disc_gen_out)

        discriminator_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        generator_grads = tape.gradient(g_loss, self.generator.trainable_variables)

        self.optimizers_discriminator_obj.apply_gradients(
            zip(discriminator_grads, self.discriminator.trainable_variables))
        self.optimizers_generator_obj.apply_gradients(zip(generator_grads, self.generator.trainable_variables))

        return d_loss, g_loss

    def train(self, dataset):
        file_writer = tf.summary.create_file_writer("./logs/{}".format(self.global_step.numpy()))
        with file_writer.as_default():
            for epoch in range(self.epochs):
                for ds in dataset:
                    d_loss, g_loss = self.train_step(ds)
                    acc = self.metrics_acc_obj.result() * 100

                    tf.summary.scalar("d_loss", step=self.check_point.global_step.numpy(), data=d_loss)
                    tf.summary.scalar("g_loss", step=self.check_point.global_step.numpy(), data=g_loss)
                    tf.summary.scalar("acc", step=self.check_point.global_step.numpy(), data=acc)

                    if self.check_point.global_step.numpy() % self.save_interval == 0:
                        self.check_point_manager.save()
                        noise = tf.random.normal(shape=(10, self.latent_dim))
                        gen_img = self.generator(noise)
                        tf.summary.image("gen_img", step=self.check_point.global_step.numpy(), data=gen_img,
                                         max_outputs=tf.shape(gen_img)[0])
                        file_writer.flush()

                    print("{}/{} global_step:{} d_loss:{} g_loss:{} acc:{}".format(epoch + 1, self.epochs,
                                                                                   self.check_point.global_step.numpy(),
                                                                                   d_loss,
                                                                                   g_loss, acc))
                    self.check_point.global_step.assign_add(1)

            file_writer.flush()


def run_main(argv):
    kwargs = {'epochs': 10,
              'latent_dim': 100, 'batch_size': 32,
              "check_point_dir": "./check_point", "save_interval": 100}
    main(**kwargs)


def main(epochs, latent_dim, batch_size, check_point_dir, save_interval):
    ds_train, ds_test = create_ds(batch_size)
    gan = Gan(epochs=epochs, latent_dim=latent_dim, batch_size=batch_size, check_point_dir=check_point_dir,
              save_interval=save_interval)
    print('Training ...')
    return gan.train(ds_train)


if __name__ == '__main__':
    app.run(run_main)
