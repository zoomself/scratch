import tensorflow as tf
import matplotlib.pyplot as plt
import anime.tf_record_creator as creator

AUTO_NUM = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
LATENT_DIM = 128
IMG_SHAPE = (96, 96, 3)
SAVE_INTERVAL = 100

ct = creator.Creator(record_by_dataset_saved_file=creator.SMALL_RECORD_FILE)
ds = ct.read().batch(BATCH_SIZE).prefetch(AUTO_NUM)


def create_generator():
    seq = tf.keras.models.Sequential([
        tf.keras.layers.Dense(24 * 24 * 128),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Reshape(target_shape=(24, 24, 128)),

        tf.keras.layers.Conv2DTranspose(64, 3, 2, padding="same"),  # (48,48,64)
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Conv2DTranspose(3, 3, 2, padding="same", activation=tf.keras.activations.tanh),
        # (96,96,3)

    ])

    input_img = tf.keras.layers.Input((LATENT_DIM,))
    output_img = seq(input_img)
    model = tf.keras.models.Model(input_img, output_img, name="generator")

    return model


def create_discriminator():
    seq = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, 3, 2, padding="same", input_shape=IMG_SHAPE),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(128, 3, 2, padding="same"),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, tf.keras.activations.sigmoid),
    ])
    input_img = tf.keras.layers.Input(IMG_SHAPE)
    output_img = seq(input_img)
    model = tf.keras.models.Model(input_img, output_img, name="discriminator")
    return model


class GAN(object):
    def __init__(self, checkpoint_dir=None):
        self.generator = create_generator()
        self.discriminator = create_discriminator()

        self.generator.summary()
        self.discriminator.summary()

        self.loss_obj = tf.keras.losses.binary_crossentropy

        self.discriminator_opt_obj = tf.keras.optimizers.Adam()
        self.generator_opt_obj = tf.keras.optimizers.Adam()

        self.generator_metrics = tf.keras.metrics.BinaryAccuracy()
        self.generator_metrics_loss = tf.keras.metrics.Mean()

        if checkpoint_dir:
            if not tf.io.gfile.exists(checkpoint_dir):
                tf.io.gfile.makedirs(checkpoint_dir)

            self.check_point = tf.train.Checkpoint(
                generator_opt=self.generator_opt_obj,
                discriminator_opt=self.discriminator_opt_obj,
                generator=self.generator,
                discriminator=self.discriminator,
                generator_metrics=self.generator_metrics,
                generator_metrics_loss=self.generator_metrics_loss,
                global_step=tf.Variable(1)
            )
            self.check_point_manager = tf.train.CheckpointManager(self.check_point, checkpoint_dir, max_to_keep=3)
            self.check_point.restore(self.check_point_manager.latest_checkpoint)
            if self.check_point_manager.latest_checkpoint:
                print("Restored from {}".format(self.check_point_manager.latest_checkpoint))
            else:
                print("Initializing from scratch.")

    def discriminator_loss(self, real_out, fake_out):
        loss_real = self.loss_obj(tf.ones_like(real_out), real_out)
        loss_gen = self.loss_obj(tf.zeros_like(real_out), fake_out)
        return loss_gen + loss_real

    def generator_loss(self, fake_out):
        return self.loss_obj(tf.ones_like(fake_out), fake_out)

    @tf.function
    def train_step(self, x):
        batch = tf.shape(x)[0]
        noise = tf.random.normal(shape=(batch, LATENT_DIM))
        with tf.GradientTape() as tape_discriminator, tf.GradientTape() as tape_generator:
            gen_images = self.generator(noise)

            fake_out = self.discriminator(gen_images)
            real_out = self.discriminator(x)

            loss_discriminator = self.discriminator_loss(real_out, fake_out)
            loss_generator = self.generator_loss(fake_out)

        self.generator_metrics_loss(loss_generator)
        self.generator_metrics(tf.ones_like(fake_out), fake_out)

        grads_discriminator = tape_discriminator.gradient(loss_discriminator, self.discriminator.trainable_variables)
        grads_generator = tape_generator.gradient(loss_generator, self.generator.trainable_variables)

        self.discriminator_opt_obj.apply_gradients(zip(grads_discriminator, self.discriminator.trainable_variables))
        self.generator_opt_obj.apply_gradients(zip(grads_generator, self.generator.trainable_variables))

    def train(self, epochs, ds_train):
        file_writer = tf.summary.create_file_writer("temp/logs")
        with file_writer.as_default():
            for e in range(epochs):
                for d in ds_train:
                    self.train_step(d)
                    print(
                        "{}/{}  global_step:{} loss:{:.4f} acc:{}".format(e + 1, epochs,
                                                                          self.check_point.global_step.numpy(),
                                                                          self.generator_metrics_loss.result(),
                                                                          self.generator_metrics.result() * 100))
                    tf.summary.scalar(name="loss", step=self.check_point.global_step.numpy(),
                                      data=self.generator_metrics_loss.result())
                    tf.summary.scalar(name="acc", step=self.check_point.global_step.numpy(),
                                      data=self.generator_metrics.result() * 100)

                    self.check_point.global_step.assign_add(1)

                    if self.check_point.global_step % SAVE_INTERVAL == 0:
                        self.check_point_manager.save()
                        max_outputs = 10
                        noise = tf.random.normal(shape=(max_outputs, LATENT_DIM))
                        predict_img = self.generator(noise)
                        tf.summary.image(name="predict_img", data=predict_img,
                                         step=self.check_point.global_step.numpy(), max_outputs=max_outputs)
                        file_writer.flush()
                        arr = tf.clip_by_value(predict_img.numpy()[0], clip_value_min=0.0, clip_value_max=1.0)
                        plt.imsave("./saved_imgs/{}.jpg".format(self.check_point.global_step.numpy()), arr.numpy())

            file_writer.flush()


gan = GAN("./check_point")
gan.train(100, ds)
