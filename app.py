from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

app = Flask(__name__)

# Load the dataset
dataset = pd.read_csv(r'C:\Users\haris\OneDrive\Documents\GitHub\PBL\heart.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Preprocess the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train the SVM model
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    time = float(request.form['time'])
    heartbeat = float(request.form['heartbeat'])

    # Check for extreme heart rate values
    if heartbeat > 200 or heartbeat < 30:
        output = "DEAD"
    else:
        # Prepare the input data
        input_data = np.array([[time, heartbeat]])
        input_data = sc.transform(input_data)

        # Make a prediction
        prediction = classifier.predict(input_data)

        # Display the output
        if prediction[0] == 0:
            output = "CALM"
        elif prediction[0] == 1 :
            output = "EXCITED"

    return render_template('index.html', prediction_text=output)

if __name__ == '__main__':
    app.run(debug=True)