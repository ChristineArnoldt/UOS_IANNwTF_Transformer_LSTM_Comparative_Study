import tensorflow as tf

class PositionalEmbeddingLayer(tf.keras.layers.Layer):

    def __init__(self, max_input_seq_len, embedding_size):
        super(PositionalEmbeddingLayer, self).__init__()

        self.max_input_seq_len = tf.cast(max_input_seq_len, dtype=tf.float32)
        self.embedding_size = tf.cast(embedding_size, dtype=tf.float32)

    def call(self, x):
        x = tf.cast(x, dtype=tf.float32)
        # Generate positional encodings
        pos_encoding = self.positional_encoding()

        # Expand dimensions to match the input shape
        pos_encoding = tf.expand_dims(pos_encoding, axis=0)

        # Repeat positional encodings for each example in the batch
        pos_encoding = tf.repeat(pos_encoding, tf.shape(x)[0], axis=0)

        # Add positional encodings to the input
        return x + pos_encoding

    def positional_encoding(self):
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
