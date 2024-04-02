import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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
        div_term = tf.pow(tf.cast(10000, dtype=tf.float32), 2 * tf.range(self.embedding_size // 2, dtype=tf.float32) / self.embedding_size)
        sinusoid_term = tf.sin(position / div_term)

        # Create positional encodings
        positional_encoding = tf.concat([sinusoid_term, tf.cos(position / div_term)], axis=-1)

        return positional_encoding

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embedding_size):
        super(TransformerBlock, self).__init__()

        self.multi_head_attention = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=embedding_size)
        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(embedding_size, activation=None)
        self.dropout2 = tf.keras.layers.Dropout(0.1)

        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        out1 = self.multi_head_attention(x, x)
        out1 = self.dropout1(out1)

        in_out = self.layernorm1(out1 + x)

        out2 = self.dense1(in_out)
        out2 = self.dense2(out2)
        out2 = self.dropout2(out2)

        out2 = self.layernorm2(out2 + in_out)
        return out2
    
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
        
        self.metrics_list = [tf.keras.metrics.Mean(name="loss"),tf.keras.metrics.Mean(name="val_loss")]
        
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
                x = layer(x,training)
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
        #tf.print("Label:", label)
        #tf.print("Output:", output)
        
        print(f"Shape Label: {tf.shape(label)}; Shape Output: {tf.shape(output)}")
        
        return {m.name : m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        
        sequence, label = data

        output = self.call(sequence, training=False)
        loss = self.compiled_loss(label, output)
        
        self.metrics[1].update_state(loss)

        return {m.name: m.result() for m in self.metrics} 

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
        model = TransformerModel(vocab_size, embedding_size, seq_length)
        # Modell kompilieren
        model.compile(optimizer='adam', loss='mse')
        # Modell trainieren
        history = model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
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
    