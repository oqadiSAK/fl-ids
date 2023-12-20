import tensorflow as tf

class Model(tf.keras.Model):
    def __init__(self, sample_shape):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(100, activation="relu", input_shape=sample_shape)
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.dense2 = tf.keras.layers.Dense(50, activation="relu")
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.dense3 = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.norm1(x)
        x = self.dense2(x)
        x = self.norm2(x)
        return self.dense3(x)