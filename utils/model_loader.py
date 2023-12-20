import typing
import tensorflow as tf

def get_model(sample_shape: typing.Tuple[int]) -> tf.keras.Model:
    inputs = tf.keras.Input(sample_shape)
    x = tf.keras.layers.Dense(100, activation="relu")(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dense(50, activation="relu")(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss="binary_crossentropy",
        metrics=["binary_accuracy"]
    )
    return model