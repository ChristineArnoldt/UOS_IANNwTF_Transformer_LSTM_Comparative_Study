import tensorflow as tf
from PositionalEmbeddingLayer import PositionalEmbeddingLayer
from TransformerBlock import TransformerBlock


class TransformerModel(tf.keras.Model):
    """
    Custom Transformer model implemented using TensorFlow Keras.

    Args:
    - vocabulary_size (int): Size of the vocabulary.
    - embedding_size (int): Size of the embedding dimension.
    - max_input_seq_len (int): Maximum input sequence length.

    Methods:
    - call(x, training=False): Method to perform forward pass through the model.
    - train_step(data): Method to perform one training step.
    - test_step(data): Method to perform one testing step.

    Returns:
    - Tensor: Output tensor after passing through the model.

    """

    def __init__(self, vocabulary_size, embedding_size, max_input_seq_len):
        """
        Initializes the TransformerModel.

        Args:
        - vocabulary_size (int): Size of the vocabulary.
        - embedding_size (int): Size of the embedding dimension.
        - max_input_seq_len (int): Maximum input sequence length.
        """
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.max_input_seq_len = max_input_seq_len

        # Initialize layers
        self.transformer_layer = TransformerBlock(self.embedding_size)
        self.positional_layer = PositionalEmbeddingLayer(self.max_input_seq_len, self.embedding_size)
        self.pooling_layer = tf.keras.layers.GlobalAveragePooling1D()
        self.output_layer = tf.keras.layers.Dense(vocabulary_size, activation=None)

        #  Metrics
        self.metrics_list = [tf.keras.metrics.Mean(name="loss")]

        # Layer list
        self.layer_list = [
            self.positional_layer,
            self.transformer_layer,
            self.pooling_layer,
            self.output_layer
        ]

    @property
    def metrics(self):
        """
        Returns the list of metrics.
        """
        return self.metrics_list

    def reset_metrics(self):
        """
        Resets the metrics.
        """
        for metric in self.metrics:
            metric.reset_state()

    @tf.function
    def call(self, x, training=False):
        """
        Performs forward pass through the model.

        Args:
        - x (Tensor): Input tensor of shape (batch_size, sequence_length).
        - training (bool): Whether the model is in training mode or not.

        Returns:
        - Tensor: Output tensor after passing through the model.
        """
        for layer in self.layer_list:
            try:
                x = layer(x, training)
            except:
                x = layer(x)

        return x

    @tf.function
    def train_step(self, data):
        """
        Performs one training step.

        Args:
        - data (tuple): Tuple containing input sequence and label.

        Returns:
        - dict: Dictionary containing updated loss metric.
        """

        sequence, label = data

        with tf.GradientTape() as tape:
            output = self.call(sequence, training=True)
            loss = self.compiled_loss(label, output)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # update loss metric
        self.metrics[0].update_state(loss)

        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        """
        Performs one testing step.

        Args:
        - data (tuple): Tuple containing input sequence and label.

        Returns:
        - dict: Dictionary containing updated loss metric.
        """
        sequence, label = data

        output = self.call(sequence, training=False)
        loss = self.compiled_loss(label, output)

        self.metrics[0].update_state(loss)

        return {m.name: m.result() for m in self.metrics}
