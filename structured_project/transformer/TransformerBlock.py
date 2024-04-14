import tensorflow as tf


class TransformerBlock(tf.keras.layers.Layer):
    """
    Custom layer implementing a single Transformer block.

    Args:
    - embedding_size (int): Size of the embedding dimension.

    Methods:
    - call(x): Method to perform forward pass through the Transformer block.

    Returns:
    - Tensor: Output tensor after passing through the Transformer block.
    """

    def __init__(self, embedding_size):
        """
        Initializes the TransformerBlock.

        Args:
        - embedding_size (int): Size of the embedding dimension.
        """
        super(TransformerBlock, self).__init__()
        self.multi_head_attention = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=embedding_size)
        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(embedding_size, activation=None)
        self.dropout2 = tf.keras.layers.Dropout(0.1)

        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        """
        Performs forward pass through the Transformer block.

        Args:
        - x (Tensor): Input tensor of shape (batch_size, sequence_length, embedding_size).

        Returns:
        - Tensor: Output tensor after passing through the Transformer block.
        """
        out1 = self.multi_head_attention(x, x)
        out1 = self.dropout1(out1)

        in_out = self.layernorm1(out1 + x)

        out2 = self.dense1(in_out)
        out2 = self.dense2(out2)
        out2 = self.dropout2(out2)

        out2 = self.layernorm2(out2 + in_out)
        return out2
