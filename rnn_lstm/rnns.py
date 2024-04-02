import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


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

        cell_state = f * prev_cell_state +  i * cell_state_candiate

        #
        # Determinating hidden state and output
        #

        o = self.output_gate_layer(concat_hidden_inputs)

        hidden_state = o * tf.math.tanh(cell_state)

        return hidden_state, [hidden_state, cell_state]
    
class RNNModel(tf.keras.Model):
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
        x= sequence
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

        return {m.name : m.result() for m in self.metrics}

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

        return {m.name : m.result() for m in self.metrics}


def stock_data(seq_length):
    df = pd.read_csv("rnn_lstm/IBM.csv", index_col='Date', parse_dates=["Date"])
    
    # Plot the training set
    #df["High"][:'2016'].plot(figsize=(16, 4), legend=True)
    # Plot the test set
    #df["High"]['2017':].plot(figsize=(16, 4), legend=True)
    #plt.legend(['Training set (Before 2017)', 'Test set (2017 and beyond)'])
    #plt.title('IBM stock price')
    #plt.show()
    
    training_set = df[:'2016'].iloc[:,1:2].values
    test_set = df['2017':].iloc[:,1:2].values
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    
    X_train = []
    y_train = []
    for i in range(seq_length, len(training_set_scaled)):
        X_train.append(training_set_scaled[i - seq_length:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    inputs = df["High"][len(df["High"]) - len(test_set) - seq_length:].values
    inputs = inputs.reshape(-1,1)
    inputs  = sc.transform(inputs)

    X_test = []
    for i in range(seq_length, len(inputs)):
        X_test.append(inputs[i-seq_length:i, 0])
        
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    return sc, X_train, y_train, X_test, test_set

def train_and_plot_loss(sequence_lengths):
    vocab_size = 1
    embedding_size = 64
    histories = {}  # Ein Dictionary zum Speichern der Verlaufshistorien
    predictions = {}
    
    
    for seq_length in sequence_lengths:
        
        sc, X_train, y_train, X_test, test_set = stock_data(seq_length)
        
        print(f"Training für Sequenzlänge: {seq_length}")
        
        # Modell erstellen
        model = RNNModel()
        # Modell kompilieren
        model.compile(optimizer='adam', loss='mean_squared_error')
        # Modell trainieren
        history = model.fit(X_train, y_train, epochs=10, batch_size=32) 
        
        histories[seq_length] = history  # Verlaufshistorie für die aktuelle Sequenzlänge speichern
        
        predicted_stock_price = model.predict(X_test)
        print(predicted_stock_price)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)
        print(predicted_stock_price)
        predictions[seq_length] = predicted_stock_price
    
    # Plotten des Verlustverlaufs für jede Sequenzlänge
    plt.figure(figsize=(10, 6))
    for seq_length, history in histories.items():
        plt.plot(history.history['loss'], label=f"Sequenzlänge {seq_length} (Train)")
        #plt.plot(history.history['val_loss'], label=f"Sequenzlänge {seq_length} (Val)")

    plt.title('Verlustverlauf für verschiedene Sequenzlängen')
    plt.xlabel('Epochen')
    plt.ylabel('Verlust')
    plt.legend()
    #plt.ylim(0,300)
    plt.show()
    
    plt.figure(figsize=(10,6))
    plt.plot(test_set, color='red',label="Real IBM Stock Price")
    for seq_length, prediction in predictions.items():
        plt.plot(prediction,label=f"predicted IBM Stock price for sequence length {seq_length}")
    plt.title("IBM Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("IBM Stock Price")
    plt.legend()
    plt.show()
    
    
    return histories[sequence_lengths[0]]  # Rückgabe der Verlaufshistorie für die erste Sequenzlänge

def create_weather_dataset_for_one_timestep_prediction(data, sequence_length,validation = False):
    """
    Function to create a dataset for a prediction task to predict the next value after the input sequence
    
    :param validation: 
    :param sequence_length: 
    :param data: 
    :return: 
    """
    
    sequences = []
    labels = []
    
    for i in range(len(data)-sequence_length-1):
        x = data[i:i+sequence_length]
        x = np.expand_dims(x, axis=-1)
        sequences.append(x)
        y = data[i+sequence_length]
        y = np.expand_dims(y, axis=-1)
        labels.append(y)
    
    print(sequences[0], labels[0])
    
    if not validation:
        return tf.data.Dataset.from_tensor_slices((sequences, labels)).shuffle(len(sequences)).batch(32).prefetch(tf.data.AUTOTUNE)
    else:
        return tf.data.Dataset.from_tensor_slices((sequences, labels)).shuffle(len(sequences)).batch(len(sequences)).prefetch(tf.data.AUTOTUNE)

def plot_stock_prediction(test,prediction):
    plt.plot(test,color='red',label="Real IBM Stock Price")
    plt.plot(prediction, color="blue",label="predicted IBM Stock price")
    plt.title("IBM Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("IBM Stock Price")
    plt.legend()
    plt.show()

# Trainieren und Plotten des Verlustverlaufs für Sequenzlängen 3, 10 und 20
history = train_and_plot_loss(sequence_lengths=[5,30,200])

print(history.history['loss'][-1])
    