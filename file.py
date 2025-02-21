import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import glob
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_and_preprocess_weather_data(directory='data/Weather'):
    """Load and preprocess weather data from multiple cities"""
    weather_files = glob.glob(os.path.join(directory, '*.csv'))
    all_weather_data = {}
    
    wind_mapping = {
        '南': 0, '南南西': 1, '南南東': 2, '西': 3, 
        '北西': 4, '南西': 5, '西北西': 6, '北': 7,
        '東': 8, '南東': 9, '北東': 10, '東北東': 11
    }
    
    for file in weather_files:
        city_name = os.path.basename(file).replace('.csv', '')
        df = pd.read_csv(file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        
        # Convert wind direction to numeric
        df['wind_direction'] = df['wind_direction'].map(wind_mapping).fillna(0)
        
        # Convert all columns to float and handle missing values
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].mean())
        
        # Add city prefix to columns
        df.columns = [f'{city_name}_{col}' for col in df.columns]
        all_weather_data[city_name] = df
    
    # Combine all weather data
    combined_weather = pd.concat(all_weather_data.values(), axis=1)
    print(f"Loaded weather data from {len(all_weather_data)} cities")
    
    # Handle any remaining NaN values
    nan_cols = combined_weather.columns[combined_weather.isna().any()].tolist()
    if nan_cols:
        print("Columns with NaN values:", nan_cols)
        combined_weather = combined_weather.fillna(method='ffill').fillna(method='bfill')
    
    return combined_weather

def prepare_data(demand_df, weather_df, sequence_length=24):
    """Prepare data for model training with additional checks"""
    # Create time features
    data = pd.DataFrame(index=demand_df.index)
    data['demand'] = demand_df['actual_performance']
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek
    data['month'] = data.index.month
    data['is_weekend'] = data.index.dayofweek.isin([5, 6]).astype(int)
    
    # Add weather features
    for col in weather_df.columns:
        data[col] = weather_df[col]
    
    # Handle any remaining missing values
    if data.isna().any().any():
        print("Handling remaining missing values...")
        data = data.fillna(data.mean())
    
    # Print data statistics
    print("\nData Statistics:")
    print(data.describe())
    
    # Handle infinite values
    if np.isinf(data.values).any():
        print("Replacing infinite values with mean values...")
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.fillna(data.mean())
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:(i + sequence_length)])
        y.append(scaled_data[i + sequence_length, 0])  # demand is first column
    
    X = np.array(X)
    y = np.array(y)
    
    # Final check for NaN values
    if np.isnan(X).any():
        print("Warning: NaN values found in X after preprocessing")
        X = np.nan_to_num(X, nan=0.0)
    if np.isnan(y).any():
        print("Warning: NaN values found in y after preprocessing")
        y = np.nan_to_num(y, nan=0.0)
    
    return X, y, scaler, data.columns

def create_model(input_shape):
    """Create an improved model architecture"""
    model = Sequential([
        tf.keras.Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='huber',
        metrics=['mae', 'mse']
    )
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    """Enhanced training function"""
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    return history

def analyze_feature_importance(model, X, y_test, feature_names):
    """Analyze feature importance using permutation importance"""
    base_mae = model.evaluate(X, y_test, verbose=0)[1]
    importance = []
    
    for i in range(X.shape[2]):
        X_temp = X.copy()
        X_temp[:, :, i] = np.random.permutation(X_temp[:, :, i])
        new_mae = model.evaluate(X_temp, y_test, verbose=0)[1]
        importance.append((feature_names[i], new_mae - base_mae))
    
    importance.sort(key=lambda x: x[1], reverse=True)
    
    plt.figure(figsize=(12, 6))
    features, scores = zip(*importance[:10])  # Top 10 features
    plt.bar(features, scores)
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 10 Most Important Features')
    plt.tight_layout()
    plt.show()

def main():
    # Print runtime information
    print(f"Current Date and Time (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Current User: {os.getlogin()}")
    
    # Load demand data
    demand_df = pd.read_csv('data/electricity_demand/demand.csv')
    demand_df['datetime'] = pd.to_datetime(demand_df['datetime'])
    demand_df.set_index('datetime', inplace=True)
    print("Demand data shape:", demand_df.shape)
    
    # Load weather data from all cities
    weather_df = load_and_preprocess_weather_data()
    print("Weather data shape:", weather_df.shape)
    
    # Prepare data
    X, y, scaler, feature_names = prepare_data(demand_df, weather_df)
    
    # Print data shapes and check for NaN values
    print("\nFinal Data Shapes:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("NaN in X:", np.isnan(X).any())
    print("NaN in y:", np.isnan(y).any())
    
    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Further split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Create and train model
    model = create_model((X.shape[1], X.shape[2]))
    print("\nModel Summary:")
    model.summary()
    
    # Train model
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Inverse transform predictions and actual values
    pred_data = np.zeros((len(y_pred), len(feature_names)))
    pred_data[:, 0] = y_pred.flatten()
    y_pred_inv = scaler.inverse_transform(pred_data)[:, 0]
    
    test_data = np.zeros((len(y_test), len(feature_names)))
    test_data[:, 0] = y_test
    y_test_inv = scaler.inverse_transform(test_data)[:, 0]
    
    # Calculate metrics
    mse = np.mean((y_pred_inv - y_test_inv) ** 2)
    mae = np.mean(np.abs(y_pred_inv - y_test_inv))
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100
    
    # Print detailed metrics
    print("\nDetailed Metrics:")
    print(f"Mean Squared Error: {mse:.2f} kW²")
    print(f"Mean Absolute Error: {mae:.2f} kW")
    print(f"Root Mean Squared Error: {rmse:.2f} kW")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    
    # Plot results with confidence intervals
    plt.figure(figsize=(15, 6))
    pred_std = np.std(y_pred_inv - y_test_inv)
    plt.fill_between(
        range(100),
        y_pred_inv[:100] - 1.96 * pred_std,
        y_pred_inv[:100] + 1.96 * pred_std,
        alpha=0.2,
        color='blue',
        label='95% Confidence Interval'
    )
    
    plt.plot(y_test_inv[:100], label='Actual', color='blue', alpha=0.7)
    plt.plot(y_pred_inv[:100], label='Predicted', color='red', alpha=0.7)
    plt.legend()
    plt.title('Electricity Demand Forecasting Results with Confidence Intervals')
    plt.xlabel('Time Steps')
    plt.ylabel('Demand (kW)')
    plt.grid(True)
    plt.show()
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss', color='blue', alpha=0.7)
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red', alpha=0.7)
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Analyze feature importance
    analyze_feature_importance(model, X_test, y_test, feature_names)

if __name__ == "__main__":
    main()