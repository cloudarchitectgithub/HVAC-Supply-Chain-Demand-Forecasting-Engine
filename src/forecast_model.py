"""
Forecasting Model using Prophet
Includes baseline comparison and model evaluation
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import os
import sys
from datetime import datetime, timedelta

# Handle different Prophet package names across environments
try:
    from prophet import Prophet
except ImportError:
    try:
        from fbprophet import Prophet
    except ImportError:
        raise ImportError("Prophet not installed. Run: pip install prophet")

# Handle imports for both direct execution and module import
try:
    from src.database import ForecastDatabase
except ImportError:
    # If running as script, add parent to path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from src.database import ForecastDatabase


def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive forecast accuracy metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = calculate_mape(y_true, y_pred)
    
    # Mean Absolute Scaled Error (MASE) - baseline is naive forecast
    naive_forecast = np.roll(y_true, 1)
    naive_mae = mean_absolute_error(y_true[1:], naive_forecast[1:])
    mase = mae / naive_mae if naive_mae > 0 else np.inf
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'MASE': mase
    }


class DemandForecastModel:
    """Prophet-based demand forecasting model"""
    
    def __init__(self, forecast_periods=90, seasonality_mode='multiplicative'):
        self.forecast_periods = forecast_periods
        self.seasonality_mode = seasonality_mode
        self.models = {}  # Store separate models for each product-region combination
        self.model_version = f"v1.0_{datetime.now().strftime('%Y%m%d')}"
    
    def prepare_prophet_data(self, df):
        """Prepare data in Prophet format (ds, y)"""
        prophet_df = df[['date', 'demand_units']].copy()
        prophet_df.columns = ['ds', 'y']
        prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
        return prophet_df
    
    def train(self, df, product_category, region):
        """Train Prophet model for specific product-region combination"""
        print(f"Training model for {product_category} - {region}...")
        
        # Filter data
        product_data = df[(df['product_category'] == product_category) & 
                         (df['region'] == region)].copy()
        
        if len(product_data) < 30:
            print(f"  Warning: Insufficient data ({len(product_data)} records), skipping...")
            return None
        
        # Prepare Prophet format
        prophet_data = self.prepare_prophet_data(product_data)
        
        # Initialize and configure Prophet
        model = Prophet(
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,  # Controls flexibility
            seasonality_prior_scale=10.0
        )
        
        # Add custom seasonalities
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.add_seasonality(name='quarterly', period=91.25, fourier_order=3)
        
        # Train model
        model.fit(prophet_data)
        
        # Store model
        key = f"{product_category}_{region}"
        self.models[key] = model
        
        return model
    
    def predict(self, product_category, region, periods=None):
        """Generate forecast for specific product-region"""
        key = f"{product_category}_{region}"
        
        if key not in self.models:
            raise ValueError(f"No trained model found for {product_category} - {region}")
        
        model = self.models[key]
        periods = periods or self.forecast_periods
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods)
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Extract only future predictions
        forecast_dates = forecast['ds'].tail(periods)
        forecast_values = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
        
        # Format output
        result = pd.DataFrame({
            'forecast_date': forecast_values['ds'],
            'product_category': product_category,
            'region': region,
            'predicted_demand': forecast_values['yhat'].values,
            'lower_bound': forecast_values['yhat_lower'].values,
            'upper_bound': forecast_values['yhat_upper'].values
        })
        
        return result
    
    def evaluate(self, df, product_category, region, test_size=30):
        """Evaluate model performance on test set"""
        key = f"{product_category}_{region}"
        
        if key not in self.models:
            return None
        
        # Get data for this product-region
        product_data = df[(df['product_category'] == product_category) & 
                         (df['region'] == region)].copy()
        product_data = product_data.sort_values('date').reset_index(drop=True)
        
        if len(product_data) < test_size + 30:
            return None
        
        # Split into train and test
        train_data = product_data.iloc[:-test_size]
        test_data = product_data.iloc[-test_size:]
        
        # Retrain on training data only
        prophet_train = self.prepare_prophet_data(train_data)
        model = Prophet(
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False
        )
        model.fit(prophet_train)
        
        # Predict on test period
        future = model.make_future_dataframe(periods=test_size)
        forecast = model.predict(future)
        predictions = forecast['yhat'].tail(test_size).values
        
        # Calculate metrics
        actual = test_data['demand_units'].values
        metrics = calculate_metrics(actual, predictions)
        
        return metrics
    
    def train_all_combinations(self, df):
        """Train models for all product-region combinations"""
        print("Training models for all product-region combinations...")
        
        combinations = df.groupby(['product_category', 'region']).size().reset_index()
        
        trained_count = 0
        for _, row in combinations.iterrows():
            product = row['product_category']
            region = row['region']
            
            try:
                model = self.train(df, product, region)
                if model is not None:
                    trained_count += 1
            except Exception as e:
                print(f"  Error training {product} - {region}: {e}")
        
        print(f"Successfully trained {trained_count} models")
        return trained_count
    
    def save_model(self, filepath):
        """Save trained models to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'models': self.models,
            'model_version': self.model_version,
            'forecast_periods': self.forecast_periods,
            'seasonality_mode': self.seasonality_mode
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained models from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.model_version = model_data.get('model_version', 'v1.0')
        self.forecast_periods = model_data.get('forecast_periods', 90)
        self.seasonality_mode = model_data.get('seasonality_mode', 'multiplicative')
        
        print(f"Model loaded from {filepath}")
        print(f"Loaded {len(self.models)} trained models")


def create_baseline_forecast(df, product_category, region, periods=90):
    """Create naive baseline forecast (simple moving average)"""
    product_data = df[(df['product_category'] == product_category) & 
                     (df['region'] == region)].copy()
    product_data = product_data.sort_values('date').reset_index(drop=True)
    
    if len(product_data) < 30:
        return None
    
    # Use 30-day moving average as baseline
    last_30_avg = product_data['demand_units'].tail(30).mean()
    
    # Generate future dates
    last_date = product_data['date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
    
    baseline = pd.DataFrame({
        'forecast_date': future_dates,
        'product_category': product_category,
        'region': region,
        'predicted_demand': last_30_avg,
        'lower_bound': last_30_avg * 0.8,
        'upper_bound': last_30_avg * 1.2
    })
    
    return baseline


if __name__ == '__main__':
    # Example usage
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.database import ForecastDatabase
    
    db = ForecastDatabase()
    df = db.load_historical_data()
    
    if df.empty:
        print("No historical data found. Please run ETL pipeline first.")
    else:
        print(f"Loaded {len(df):,} historical records")
        
        # Train model
        model = DemandForecastModel(forecast_periods=90)
        model.train_all_combinations(df)
        
        # Save model
        model.save_model('models/prophet_model.pkl')
        
        # Example forecast
        products = df['product_category'].unique()[:2]
        regions = df['region'].unique()[:2]
        
        for product in products:
            for region in regions:
                try:
                    forecast = model.predict(product, region)
                    print(f"\nForecast for {product} - {region}:")
                    print(forecast.head())
                except:
                    pass

