import tensorflow as tf


class TdcvModel(tf.keras.Model):
    def __init__(self):
        super(TdcvModel, self).__init__()
        self.C1 = tf.layers.Conv2D(16, 8, strides=2, padding='same', activation=tf.nn.relu)
        self.M1 = tf.layers.MaxPooling2D((2, 2), (2, 2), padding='same')
        self.C2 = tf.layers.Conv2D(7, 5, strides=2, padding='same', activation=tf.nn.relu)
        self.M2 = tf.layers.MaxPooling2D((2, 2), (2, 2), padding='same')
        self.F0 = tf.layers.Flatten()
        self.D1 = tf.layers.Dense(256)
        self.D2 = tf.layers.Dense(16)  # Descriptor Size

    def call(self, images):
        """Run the model."""
        x = self.C1(images)  # batch_size - 32x32x16
        x = self.M1(x)  # batch_size - 32x32x16
        x = self.C2(x)  # batch_size - 16x16x16
        x = self.M2(x)  # batch_size - 8x8x7
        x = self.F0(x)  # batch_size - 4x4x7
        x = self.D1(x)  # batch_size - 256
        x = self.D2(x)  # batch_size - 16
        return x