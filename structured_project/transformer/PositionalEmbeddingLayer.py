import tensorflow as tf


class PositionalEmbeddingLayer(tf.keras.layers.Layer):
    """
        Custom layer for adding positional embeddings to the input sequence.

        Args:
        - max_input_seq_len (int): Maximum length of the input sequence.
        - embedding_size (int): Size of the embedding dimension.

        Methods:
        - call(x): Method to add positional encodings to the input tensor.
        - positional_encoding(): Method to generate positional encodings.

        Returns:
        - Tensor: Input tensor with positional encodings added.
        """

    def __init__(self, max_input_seq_len, embedding_size):
        """
        Initializes the PositionalEmbeddingLayer.

        Args:
        - max_input_seq_len (int): Maximum length of the input sequence.
        - embedding_size (int): Size of the embedding dimension.
        """
        super(PositionalEmbeddingLayer, self).__init__()
        self.max_input_seq_len = tf.cast(max_input_seq_len, dtype=tf.float32)
        self.embedding_size = tf.cast(embedding_size, dtype=tf.float32)

    def call(self, x):
        """
        Adds positional encodings to the input tensor.

        Args:
        - x (Tensor): Input tensor of shape (batch_size, sequence_length, embedding_size).

        Returns:
        - Tensor: Input tensor with positional encodings added.
        """
        x = tf.cast(x, dtype=tf.float32)
        pos_encoding = self.positional_encoding()  # Generate positional encodings
        pos_encoding = tf.expand_dims(pos_encoding, axis=0)  # Expand dimensions to match the input shape
        pos_encoding = tf.repeat(pos_encoding, tf.shape(x)[0], axis=0)  # Repeat positional encodings for each example in the batch
        return x + pos_encoding  # Add positional encodings to the input tensor

    def positional_encoding(self):
        """
        Generates positional encodings.

        Returns:
        - Tensor: Positional encodings tensor of shape (max_input_seq_len, embedding_size).
        """

        # Calculate positional encodings
        position = tf.range(self.max_input_seq_len, dtype=tf.float32)
        position = tf.expand_dims(position, 1)
        position = tf.cast(position, dtype=tf.float32)
        div_term = tf.pow(tf.cast(10000, dtype=tf.float32),
                          2 * tf.range(self.embedding_size // 2, dtype=tf.float32) / self.embedding_size)
        sinusoid_term = tf.sin(position / div_term)

        # Create positional encodings
        positional_encoding = tf.concat([sinusoid_term, tf.cos(position / div_term)], axis=-1)

        return positional_encoding
