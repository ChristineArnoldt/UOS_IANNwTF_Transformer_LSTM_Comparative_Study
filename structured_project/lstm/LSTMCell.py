import tensorflow as tf


class LSTMCell(tf.keras.layers.AbstractRNNCell):
    """
    Custom LSTM cell implemented using TensorFlow Keras.

    Args:
    - units (int): Dimensionality of the output space.

    Methods:
    - get_initial_state(inputs=None, batch_size=None, dtype=None): Method to get the initial cell state.
    - call(inputs, states): Method to perform the forward pass through the cell.

    Attributes:
    - state_size: Size of the cell state.
    - output_size: Size of the output.
    """
    def __init__(self, units, **kwargs):
        """
        Initializes the LSTM_Cell.

        Args:
        - units (int): Dimensionality of the output space.
        """
        super().__init__(**kwargs)
        self.units = units

        # Define layers for gates and cell state candidate
        self.forget_gate_layer = tf.keras.layers.Dense(units=units, activation="sigmoid")
        self.input_gate_layer = tf.keras.layers.Dense(units=units, activation="sigmoid")
        self.output_gate_layer = tf.keras.layers.Dense(units=units, activation="sigmoid")
        self.cell_state_candidate_layer = tf.keras.layers.Dense(units=units, activation="tanh")

    @property
    def state_size(self):
        """
        Returns the size of the cell state.

        Returns:
        - list: List containing the size of the cell state.
        """
        return [tf.TensorShape(self.units), tf.TensorShape(self.units)]

    @property
    def output_size(self):
        """
        Returns the size of the output.

        Returns:
        - list: List containing the size of the output.
        """
        return [tf.TensorShape(self.units), tf.TensorShape(self.units)]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """
        Returns the initial cell state.

        Args:
        - inputs: Input tensor.
        - batch_size (int): Size of the batch.
        - dtype: Data type of the initial state.

        Returns:
        - list: List containing the initial hidden state and cell state.
        """
        return [tf.zeros(shape=(batch_size, self.units)), tf.zeros(shape=(batch_size, self.units))]

    def call(self, inputs, states):
        """
        Performs the forward pass through the LSTM cell.

        Args:
        - inputs: Input tensor.
        - states: List containing the previous hidden state and cell state.

        Returns:
        - hidden_state: Updated hidden state.
        - new_states: List containing the updated hidden state and cell state.
        """
        prev_hidden_state = states[0]
        prev_cell_state = states[1]

        # Concatenate previous hidden state and current input
        concat_hidden_inputs = tf.concat([prev_hidden_state, inputs], axis=-1)

        # Compute forget gate, input gate, and cell state candidate
        f = self.forget_gate_layer(concat_hidden_inputs)
        i = self.input_gate_layer(concat_hidden_inputs)
        cell_state_candidate = self.cell_state_candidate_layer(concat_hidden_inputs)

        # Update cell state
        cell_state = f * prev_cell_state + i * cell_state_candidate

        # Compute output gate
        o = self.output_gate_layer(concat_hidden_inputs)

        # Compute new hidden state
        hidden_state = o * tf.math.tanh(cell_state)

        return hidden_state, [hidden_state, cell_state]