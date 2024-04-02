import tensorflow as tf

class LSTM_Cell(tf.keras.layers.AbstractRNNCell):

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)

        self.units = units

        self.forget_gate_layer = tf.keras.layers.Dense(units=units, activation="sigmoid")
        self.input_gate_layer = tf.keras.layers.Dense(units=units, activation="sigmoid")
        self.output_gate_layer = tf.keras.layers.Dense(units=units, activation="sigmoid")

        self.cell_state_candiate_layer = tf.keras.layers.Dense(units=units, activation="tanh")

    @property
    def state_size(self):
        return [tf.TensorShape(self.units), tf.TensorShape(self.units)]

    @property
    def output_size(self):
        return [tf.TensorShape(self.units), tf.TensorShape(self.units)]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        # hidden_state, cell_state
        return [tf.zeros(shape=(batch_size, self.units)), tf.zeros(shape=(batch_size, self.units))]

    def call(self, inputs, states):
        prev_hidden_state = states[0]
        prev_cell_state = states[1]

        concat_hidden_inputs = tf.concat([prev_hidden_state, inputs], axis=-1)

        #
        # Preparing
        #

        f = self.forget_gate_layer(concat_hidden_inputs)
        i = self.input_gate_layer(concat_hidden_inputs)

        cell_state_candiate = self.cell_state_candiate_layer(concat_hidden_inputs)

        #
        # Update cell state
        #

        cell_state = f * prev_cell_state + i * cell_state_candiate

        #
        # Determinating hidden state and output
        #

        o = self.output_gate_layer(concat_hidden_inputs)

        hidden_state = o * tf.math.tanh(cell_state)

        return hidden_state, [hidden_state, cell_state]