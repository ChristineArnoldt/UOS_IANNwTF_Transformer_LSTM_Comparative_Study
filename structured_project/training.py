import matplotlib.pyplot as plt
from structured_project.lstm.LSTMModel import LSTMModel
from structured_project.preprocessing import stock_data
from structured_project.transformer.TransformerModel import TransformerModel
from sklearn.metrics import mean_absolute_error, mean_squared_error


def train_and_plot_loss(sequence_lengths):
    vocab_size = 1
    embedding_size = 64
    predictions_lstm = {}
    predictions_transformer = {}

    for seq_length in sequence_lengths:
        sc, X_train, y_train, X_test, test_set = stock_data(seq_length)

        print(f"Training für Sequenzlänge: {seq_length}")

        transformer = TransformerModel(vocab_size, embedding_size, seq_length)
        transformer.compile(optimizer='adam', loss='mse')
        transformer.fit(X_train, y_train, epochs=10, batch_size=32)

        predicted_stock_price_transformer = transformer.predict(X_test)
        predicted_stock_price_transformer = sc.inverse_transform(predicted_stock_price_transformer)
        predictions_transformer[seq_length] = predicted_stock_price_transformer

        lstm = LSTMModel()
        lstm.compile(optimizer='adam', loss='mean_squared_error')
        lstm.fit(X_train, y_train, epochs=10, batch_size=32)

        predicted_stock_price_lstm = lstm.predict(X_test)
        predicted_stock_price_lstm = sc.inverse_transform(predicted_stock_price_lstm)
        predictions_lstm[seq_length] = predicted_stock_price_lstm

        # Berechnen der Genauigkeit
        mae_transformer = mean_absolute_error(test_set, predicted_stock_price_transformer)
        mse_transformer = mean_squared_error(test_set, predicted_stock_price_transformer)
        mae_lstm = mean_absolute_error(test_set, predicted_stock_price_lstm)
        mse_lstm = mean_squared_error(test_set, predicted_stock_price_lstm)

        # Ausgabe der Genauigkeit
        print(f"Sequenzlänge: {seq_length}")
        print(f"Transformer - MAE: {mae_transformer}, MSE: {mse_transformer}")
        print(f"LSTM - MAE: {mae_lstm}, MSE: {mse_lstm}")

    # Plotten des Verlustverlaufs für jede Sequenzlänge
    '''
    plt.figure(figsize=(10, 6))
    for seq_length, history in histories.items():
        plt.plot(history.history['loss'], label=f"Sequenzlänge {seq_length} (Train)")

    plt.title('Verlustverlauf für verschiedene Sequenzlängen')
    plt.xlabel('Epochen')
    plt.ylabel('Verlust')
    plt.legend()
    plt.show()
    '''

    plt.figure(figsize=(10, 6))
    plt.plot(test_set, color='red', label="Real IBM Stock Price")
    for seq_length, prediction in predictions_lstm.items():
        plt.plot(prediction, label=f"LSTM - Sequenzlänge {seq_length}")
    for seq_length, prediction in predictions_transformer.items():
        plt.plot(prediction, label=f"Transformer - Sequenzlänge {seq_length}")
    plt.title("IBM Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("IBM Stock Price")
    plt.legend()
    plt.show()


# Trainieren und Plotten des Verlustverlaufs für Sequenzlängen 3, 10 und 20
train_and_plot_loss(sequence_lengths=[5,400])
