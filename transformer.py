import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class EmbeddTokenAndPosLayer(tf.keras.layers.Layer):
     
    def __init__(self, max_input_seq_len, embedding_size):
        super(EmbeddTokenAndPosLayer, self).__init__()

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
    
class TokenPredictor(tf.keras.Model):
    def __init__(self, vocabulary_size, embedding_size, max_input_seq_len):
        super(TokenPredictor, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.max_input_seq_len = max_input_seq_len
        
        self.transformer_layer = TransformerBlock(self.embedding_size)
        self.positional_layer = EmbeddTokenAndPosLayer(self.max_input_seq_len, self.embedding_size)
        self.pooling_layer = tf.keras.layers.GlobalAveragePooling1D()
        self.output_layer = tf.keras.layers.Dense(vocabulary_size, activation=None)
        
        self.metrics_list = [tf.keras.metrics.Mean(name="loss")]
        
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
            target_token = tf.expand_dims(label, -1)
            loss = self.compiled_loss(label, output)
            
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # update loss metric
        self.metrics[0].update_state(loss)
        
        return {m.name : m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        
        sequence, label = data
        target_token = tf.expand_dims(label, -1)

        output = self.call(sequence, training=False)
        loss = self.compiled_loss(target_token, output)
        
        self.metrics[0].update_state(loss)

        return {m.name: m.result() for m in self.metrics} 
    
def create_dataset(sequence_length):
    X_train = tf.cast(np.random.randint(0, 10, (1024, sequence_length)), tf.float32)
    X_train = tf.expand_dims(X_train, axis=-1)
    y_train = tf.reduce_sum(X_train[:, :, 0:2], axis=1)  # Addition der ersten beiden Zahlen in jeder Sequenz
    y_train = tf.cast(y_train, tf.float32)

    X_val = tf.cast(np.random.randint(0, 10, (512, sequence_length)), tf.float32)
    X_val = tf.expand_dims(X_val, axis=-1)
    y_val = tf.reduce_sum(X_val[:, :, 0:2], axis=1)  # Addition der ersten beiden Zahlen in jeder Sequenz
    y_val = tf.cast(y_val, tf.float32)
    
    return X_train, y_train, X_val, y_val

def train_and_plot_loss(sequence_lengths):
    vocab_size = 6000
    embedding_size = 64
    histories = {}  # Ein Dictionary zum Speichern der Verlaufshistorien
    
    for seq_length in sequence_lengths:
        print(f"Training für Sequenzlänge: {seq_length}")
        
        # Erstellen des Datensets für die aktuelle Sequenzlänge
        X_train, y_train, X_val, y_val = create_dataset(seq_length)
        
        # Modell erstellen
        model = TokenPredictor(vocab_size, embedding_size, seq_length)
        
        
        # Modell kompilieren
        model.compile(optimizer='adam', loss='mse')
        
        # Modell trainieren
        history = model.fit(X_train, y_train, 
                            epochs=100, 
                            batch_size=32, 
                            validation_data=(X_val, y_val),
                            verbose=0)  # Verbosity auf 0 setzen, um den Trainingsfortschritt nicht auszugeben
        
        histories[seq_length] = history  # Verlaufshistorie für die aktuelle Sequenzlänge speichern
    
    # Plotten des Verlustverlaufs für jede Sequenzlänge
    plt.figure(figsize=(10, 6))
    for seq_length, history in histories.items():
        plt.plot(history.history['loss'], label=f"Sequenzlänge {seq_length} (Train)")
        # plt.plot(history.history['val_loss'], label=f"Sequenzlänge {seq_length} (Val)")

    plt.title('Verlustverlauf für verschiedene Sequenzlängen')
    plt.xlabel('Epochen')
    plt.ylabel('Verlust')
    plt.legend()
    plt.ylim(0,300)
    plt.show()
    
    return histories[sequence_lengths[0]]  # Rückgabe der Verlaufshistorie für die erste Sequenzlänge

# Trainieren und Plotten des Verlustverlaufs für Sequenzlängen 3, 10 und 20
history = train_and_plot_loss(sequence_lengths=[5,10,15,20,25,30])

print(history.history['loss'][-1])