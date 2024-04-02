import tensorflow as tf

from structured_project.lstm.LSTMCell import LSTM_Cell


class LSTMModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.layer_list = [
            tf.keras.layers.RNN(LSTM_Cell(units=100), return_sequences=True),
            tf.keras.layers.Dropout(rate=0.3),
            tf.keras.layers.RNN(LSTM_Cell(units=80), return_sequences=True),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.RNN(LSTM_Cell(units=50), return_sequences=True),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.RNN(LSTM_Cell(units=30)),
            tf.keras.layers.Dropout(rate=0.3),
        ]

        self.output_layer = tf.keras.layers.Dense(units=1)

        self.metrics_list = [tf.keras.metrics.Mean(name="loss")]

    @property
    def metrics(self):
        return self.metrics_list

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_state()

    @tf.function
    def call(self, sequence, training=False):
        x = sequence
        for layer in self.layer_list:
            x = layer(x)
        x = self.output_layer(x)
        return x

    @tf.function
    def train_step(self, data):
        """
        Standard train_step method
        :param data:
        :return:
        """

        sequence, label = data
        with tf.GradientTape() as tape:
            output = self.call(sequence, training=True)
            loss = self.compiled_loss(label, output, regularization_losses=self.losses)
        gradients = tape.gradient(loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.metrics[0].update_state(loss)

        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        """
        Standard test_step method
        :param data:
        :return:
        """

        sequence, label = data
        output = self.call(sequence, training=False)
        loss = self.compiled_loss(label, output, regularization_losses=self.losses)

        self.metrics[0].update_state(loss)

        return {m.name: m.result() for m in self.metrics}
