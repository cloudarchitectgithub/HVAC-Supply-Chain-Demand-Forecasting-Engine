"""
Database utilities for storing and retrieving forecast data
Uses SQLite for simplicity, but designed to be easily migrated to enterprise databases
"""

import sqlite3
import pandas as pd
from sqlalchemy import create_engine, text
import os
from datetime import datetime


class ForecastDatabase:
    """Manages database operations for the forecasting system"""
    
    def __init__(self, db_path='data/forecast_db.sqlite'):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.engine = create_engine(f'sqlite:///{db_path}', echo=False)
        self._create_tables()
    
    def _create_tables(self):
        """Create database schema if it doesn't exist"""
        with self.engine.connect() as conn:
            # Historical demand data
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS historical_demand (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    product_category TEXT NOT NULL,
                    region TEXT NOT NULL,
                    demand_units INTEGER NOT NULL,
                    revenue_usd REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, product_category, region)
                )
            """))
            
            # Forecasts
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS forecasts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    forecast_date DATE NOT NULL,
                    product_category TEXT NOT NULL,
                    region TEXT NOT NULL,
                    predicted_demand REAL NOT NULL,
                    lower_bound REAL,
                    upper_bound REAL,
                    model_version TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # Model metrics
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_version TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    product_category TEXT,
                    region TEXT,
                    evaluation_date DATE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # Model training history
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS model_training_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_version TEXT NOT NULL,
                    training_date DATE NOT NULL,
                    training_samples INTEGER,
                    model_type TEXT,
                    status TEXT,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            conn.commit()
    
    def load_historical_data(self, product_category=None, region=None, start_date=None, end_date=None):
        """Load historical demand data with optional filters"""
        query = "SELECT * FROM historical_demand WHERE 1=1"
        params = {}
        
        if product_category:
            query += " AND product_category = :product_category"
            params['product_category'] = product_category
        if region:
            query += " AND region = :region"
            params['region'] = region
        if start_date:
            query += " AND date >= :start_date"
            params['start_date'] = start_date
        if end_date:
            query += " AND date <= :end_date"
            params['end_date'] = end_date
        
        query += " ORDER BY date ASC"
        
        return pd.read_sql_query(text(query), self.engine.connect(), params=params)
    
    def save_historical_data(self, df):
        """Save historical demand data to database"""
        df.to_sql('historical_demand', self.engine, if_exists='append', index=False, method='multi')
    
    def save_forecast(self, forecast_df, model_version='v1.0'):
        """Save forecast predictions to database"""
        forecast_df['model_version'] = model_version
        forecast_df.to_sql('forecasts', self.engine, if_exists='append', index=False)
    
    def get_latest_forecast(self, product_category=None, region=None):
        """Get the most recent forecast for given filters"""
        query = """
            SELECT * FROM forecasts 
            WHERE forecast_date = (SELECT MAX(forecast_date) FROM forecasts)
        """
        params = {}
        
        if product_category:
            query += " AND product_category = :product_category"
            params['product_category'] = product_category
        if region:
            query += " AND region = :region"
            params['region'] = region
        
        query += " ORDER BY forecast_date ASC"
        
        return pd.read_sql_query(text(query), self.engine.connect(), params=params)
    
    def save_model_metrics(self, metrics_dict, model_version='v1.0', product_category=None, region=None):
        """Save model performance metrics"""
        metrics_data = []
        for metric_name, metric_value in metrics_dict.items():
            metrics_data.append({
                'model_version': model_version,
                'metric_name': metric_name,
                'metric_value': metric_value,
                'product_category': product_category,
                'region': region,
                'evaluation_date': datetime.now().date()
            })
        
        df_metrics = pd.DataFrame(metrics_data)
        df_metrics.to_sql('model_metrics', self.engine, if_exists='append', index=False)
    
    def get_model_metrics(self, model_version=None):
        """Retrieve model performance metrics"""
        query = "SELECT * FROM model_metrics WHERE 1=1"
        params = {}
        
        if model_version:
            query += " AND model_version = :model_version"
            params['model_version'] = model_version
        
        query += " ORDER BY evaluation_date DESC, created_at DESC"
        
        return pd.read_sql_query(text(query), self.engine.connect(), params=params)
    
    def log_training(self, model_version, training_samples, model_type, status='completed', notes=None):
        """Log model training event"""
        training_data = {
            'model_version': model_version,
            'training_date': datetime.now().date(),
            'training_samples': training_samples,
            'model_type': model_type,
            'status': status,
            'notes': notes
        }
        
        df = pd.DataFrame([training_data])
        df.to_sql('model_training_history', self.engine, if_exists='append', index=False)

