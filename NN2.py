import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.1
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.1

        self.bias_hidden = np.zeros(hidden_size)
        self.bias_output = np.zeros(output_size)

    def sigmoid(self, x):
        return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))


    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def rmse_loss(self, y_true, y_pred):
        return np.sqrt(np.mean(np.square(y_true - y_pred)))

    def forward(self, X):
        hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(hidden_input)
        output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        predicted_output = self.sigmoid(output_input)
        return predicted_output

    def backward(self, X, y, predicted_output, learning_rate):
        error = y - predicted_output
        delta_output = error * self.sigmoid_derivative(predicted_output)
        error_hidden = np.dot(delta_output, self.weights_hidden_output.T)
        delta_hidden = error_hidden * self.sigmoid_derivative(self.hidden_output)
        self.weights_hidden_output += np.dot(self.hidden_output.T, delta_output) * learning_rate
        self.weights_input_hidden += np.dot(X.T, delta_hidden) * learning_rate
        self.bias_output += np.sum(delta_output, axis=0) * learning_rate
        self.bias_hidden += np.sum(delta_hidden, axis=0) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            predicted_output = self.forward(X)
            self.backward(X, y, predicted_output, learning_rate)
            loss = self.rmse_loss(y, predicted_output)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss:.4f}')

data = pd.read_csv('aggregated_data.csv')

data['Date'] = pd.to_datetime(data['Date'])
data['Jour'] = data['Date'].dt.day
data['Mois'] = data['Date'].dt.month
data['Année'] = data['Date'].dt.year
data['Heure'] = data['Date'].dt.hour
data['Minutes'] = data['Date'].dt.minute

encoder = OneHotEncoder()
city_encoded = encoder.fit_transform(data[['City']])
city_df = pd.DataFrame(city_encoded.toarray(), columns=encoder.get_feature_names_out(['City']))

feature_cols = ['Longitude', 'Latitude', 'Jour', 'Mois', 'Année', 'Heure', 'Minutes'] + list(city_df.columns)

X = pd.concat([data[feature_cols[:-len(city_df.columns)]], city_df], axis=1)
y = data['Value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = y_train.to_numpy().reshape(-1, 1)
y_test = y_test.to_numpy().reshape(-1, 1)


input_size = X_train.shape[1]
hidden_size = 100
output_size = 1
learning_rate = 0.1
epochs = 50

nn = NeuralNetwork(input_size, hidden_size, output_size)

nn.train(X_train, y_train, epochs, learning_rate)

predictions = nn.forward(X)
print("Prédictions:")
predictions_df = pd.DataFrame(predictions, columns=['Predicted_Value'])
predictions_df.to_csv('predictions.csv', index=False)
print("Les prédictions ont été sauvegardées dans 'predictions.csv'")