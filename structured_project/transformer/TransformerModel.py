import tensorflow as tf
from structured_project.transformer.PositionalEmbeddingLayer import PositionalEmbeddingLayer
from structured_project.transformer.TransformerBlock import TransformerBlock


class TransformerModel(tf.keras.Model):
    def __init__(self, vocabulary_size, embedding_size, max_input_seq_len):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.max_input_seq_len = max_input_seq_len

        self.transformer_layer = TransformerBlock(self.embedding_size)
        self.positional_layer = PositionalEmbeddingLayer(self.max_input_seq_len, self.embedding_size)
        self.pooling_layer = tf.keras.layers.GlobalAveragePooling1D()
        self.output_layer = tf.keras.layers.Dense(vocabulary_size, activation=None)

        self.metrics_list = [tf.keras.metrics.Mean(name="loss"), tf.keras.metrics.Mean(name="val_loss")]

        self.layer_list = [
            self.positional_layer,
            self.transformer_layer,
            self.pooling_layer,
            self.output_layer
        ]

    @property
    def metrics(self):
        return self.metrics_list

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_state()

    @tf.function
    def call(self, x, training=False):
        for layer in self.layer_list:
            try:
                x = layer(x, training)
            except:
                x = layer(x)

        return x

    @tf.function
    def train_step(self, data):

        sequence, label = data

        with tf.GradientTape() as tape:
            output = self.call(sequence, training=True)
            loss = self.compiled_loss(label, output)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # update loss metric
        self.metrics[0].update_state(loss)
        # tf.print("Label:", label)
        # tf.print("Output:", output)

        print(f"Shape Label: {tf.shape(label)}; Shape Output: {tf.shape(output)}")

        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):

        sequence, label = data

        output = self.call(sequence, training=False)
        loss = self.compiled_loss(label, output)

        self.metrics[1].update_state(loss)

        return {m.name: m.result() for m in self.metrics}
