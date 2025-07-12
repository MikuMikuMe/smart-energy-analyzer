Building a web-based tool for analyzing and optimizing household energy consumption involves a few components: data collection, machine learning modeling, and a web interface. Below is a simplified version of such a program using Python with Flask for the web interface, Pandas for data handling, and scikit-learn for machine learning. This version assumes you have energy consumption data stored in a CSV file.

Ensure to have all necessary libraries installed:
```bash
pip install flask pandas scikit-learn
```

Here's a step-by-step implementation:

```python
from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import traceback

app = Flask(__name__)

# Load and prepare data
def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        # Assuming 'features' and 'target' are columns in the CSV
        X = data.drop('target', axis=1)
        y = data['target']
        return X, y
    except Exception as e:
        print(f"Failed to load data: {e}")
        print(traceback.format_exc())
        return None, None

# Train a simple model
def train_model(X, y):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print(f"Model performance (MSE): {mse}")
        
        return model
    except Exception as e:
        print(f"Failed to train model: {e}")
        print(traceback.format_exc())
        return None

# Provide recommendations based on the model
def analyze_energy(data, model):
    try:
        prediction = model.predict(data)
        # Simple recommendation method
        recommendation = "Consider reducing usage" if prediction.mean() > 100 else "Usage is optimal"
        return prediction, recommendation
    except Exception as e:
        print(f"Failed to analyze energy: {e}")
        print(traceback.format_exc())
        return None, "Error in analysis"

# Index route
@app.route('/')
def home():
    return render_template('index.html') # Create an index.html file for home page

# API route for analysis
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        input_data = request.json['data']
        input_df = pd.DataFrame([input_data])
        
        prediction, recommendation = analyze_energy(input_df, model)
        
        return jsonify({
            "prediction": prediction.tolist(),
            "recommendation": recommendation
        })
    except Exception as e:
        print(f"Error in /analyze route: {e}")
        print(traceback.format_exc())
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == '__main__':
    # Load data
    features, target = load_data('energy_data.csv')
    if features is not None and target is not None:
        # Train model
        model = train_model(features, target)
        if model:
            app.run(debug=True)
        else:
            print("Failed to train the model.")
    else:
        print("Failed to load the data.")
```

### Key Components:

1. **Data Loading:**
   - Loads household energy data from a CSV file.
   - Expects a column named 'target' for the values to predict (e.g., energy consumption) and the rest as features.

2. **Model Training:**
   - Trains a Random Forest regressor to predict energy consumption.

3. **Error Handling:**
   - Uses try-except blocks to catch and print errors for debugging.

4. **Flask Endpoint:**
   - `/` for the home page, which is expected to render `index.html`.
   - `/analyze` API endpoint accepts POST requests with JSON data for prediction.

5. **Recommendations:**
   - A simple threshold-based recommendation system.

Note: Extend this example by integrating real-time or more complex data, adding further analysis, improving error handling, and creating a detailed HTML user interface with more insights and visualization options.