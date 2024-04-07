import tensorflow as tf

from lstm.LSTMCell import LSTMCell


class LSTMModel(tf.keras.Model):
    """Custom LSTM model class."""
    def __init__(self):
        """Initialize the LSTM model."""
        super().__init__()

        # Define the layers of the model
        self.layer_list = [
            tf.keras.layers.RNN(LSTMCell(units=72), return_sequences=True),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.RNN(LSTMCell(units=30), return_sequences=True),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.RNN(LSTMCell(units=20)),
            tf.keras.layers.Dropout(rate=0.3),
        ]
        self.output_layer = tf.keras.layers.Dense(units=1)
        self.metrics_list = [tf.keras.metrics.Mean(name="loss")]

    @property
    def metrics(self):
        """Return the list of metrics."""
        return self.metrics_list

    def reset_metrics(self):
        """Reset the state of the metrics."""
        for metric in self.metrics:
            metric.reset_state()

    @tf.function
    def call(self, sequence, training=False):
        """
        Apply the layers sequentially to the input sequence and return the output prediction.

        Args:
            sequence (tf.Tensor): Input sequence data.
            training (bool): Flag indicating whether the model is in training mode.

        Returns:
            tf.Tensor: Output prediction.
        """
        x = sequence
        for layer in self.layer_list:
            x = layer(x)
        x = self.output_layer(x)
        return x

    @tf.function
    def train_step(self, data):
        """
        Perform a single training step.

        Args:
            data (tuple): Tuple containing input sequence and corresponding labels.

        Returns:
            dict: Dictionary containing the training metrics.
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
        Perform a single testing step.

        Args:
            data (tuple): Tuple containing input sequence and corresponding labels.

        Returns:
            dict: Dictionary containing the testing metrics.
        """

        sequence, label = data
        output = self.call(sequence, training=False)
        loss = self.compiled_loss(label, output, regularization_losses=self.losses)

        self.metrics[0].update_state(loss)

        return {m.name: m.result() for m in self.metrics}
