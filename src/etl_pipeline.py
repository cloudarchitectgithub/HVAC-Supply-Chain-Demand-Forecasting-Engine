"""
ETL Pipeline for Supply Chain Demand Forecasting
Extract, Transform, and Load data from raw sources to processed database
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Handle imports for both direct execution and module import
try:
    from src.database import ForecastDatabase
except ImportError:
    # If running as script, add parent to path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from src.database import ForecastDatabase


class ETLPipeline:
    """Handles Extract, Transform, Load operations for demand forecasting"""
    
    def __init__(self, db_path='data/forecast_db.sqlite'):
        self.db = ForecastDatabase(db_path)
    
    def extract(self, source_path):
        """
        Extract data from source (CSV, ERP system, etc.)
        In production, this would connect to ERP/SCE systems via API
        """
        print(f"Extracting data from {source_path}...")
        
        if source_path.endswith('.csv'):
            df = pd.read_csv(source_path)
            df['date'] = pd.to_datetime(df['date'])
            return df
        else:
            raise ValueError(f"Unsupported file format: {source_path}")
    
    def transform(self, df):
        """
        Transform raw data into features suitable for forecasting
        Feature engineering: lags, rolling averages, seasonality indicators
        """
        print("Transforming data with feature engineering...")
        
        # Sort by date
        df = df.sort_values(['date', 'product_category', 'region']).reset_index(drop=True)
        
        # Aggregate daily totals (in case of multiple entries per day)
        df_agg = df.groupby(['date', 'product_category', 'region']).agg({
            'demand_units': 'sum',
            'revenue_usd': 'sum'
        }).reset_index()
        
        # Feature engineering for each product-region combination
        transformed_data = []
        
        for (product, region), group in df_agg.groupby(['product_category', 'region']):
            group = group.sort_values('date').reset_index(drop=True)
            
            # Lag features (previous day, week, month)
            group['demand_lag_1'] = group['demand_units'].shift(1)
            group['demand_lag_7'] = group['demand_units'].shift(7)
            group['demand_lag_30'] = group['demand_units'].shift(30)
            
            # Rolling averages
            group['demand_ma_7'] = group['demand_units'].rolling(window=7, min_periods=1).mean()
            group['demand_ma_30'] = group['demand_units'].rolling(window=30, min_periods=1).mean()
            group['demand_ma_90'] = group['demand_units'].rolling(window=90, min_periods=1).mean()
            
            # Rolling standard deviation (volatility)
            group['demand_std_7'] = group['demand_units'].rolling(window=7, min_periods=1).std()
            group['demand_std_30'] = group['demand_units'].rolling(window=30, min_periods=1).std()
            
            # Time-based features
            group['year'] = group['date'].dt.year
            group['month'] = group['date'].dt.month
            group['day_of_year'] = group['date'].dt.dayofyear
            # FIXED: Handle isocalendar week properly
            group['week_of_year'] = group['date'].dt.isocalendar().week.astype(int)
            group['day_of_week'] = group['date'].dt.dayofweek
            group['is_weekend'] = (group['day_of_week'] >= 5).astype(int)
            group['quarter'] = group['date'].dt.quarter
            
            # Seasonal indicators
            group['is_summer'] = group['month'].isin([6, 7, 8]).astype(int)
            group['is_winter'] = group['month'].isin([12, 1, 2]).astype(int)
            
            # Growth rate (week-over-week, month-over-month)
            group['wow_growth'] = group['demand_units'].pct_change(periods=7)
            group['mom_growth'] = group['demand_units'].pct_change(periods=30)
            
            # FIXED: Use bfill() instead of deprecated fillna(method='bfill')
            group = group.bfill().fillna(0)
            
            transformed_data.append(group)
        
        df_transformed = pd.concat(transformed_data, ignore_index=True)
        
        print(f"Transformed {len(df_transformed):,} records with {len(df_transformed.columns)} features")
        return df_transformed
    
    def load(self, df, table_name='historical_demand'):
        """
        Load transformed data into database
        """
        print(f"Loading data into {table_name}...")
        
        # Prepare data for database (only keep essential columns for historical_demand table)
        df_to_load = df[['date', 'product_category', 'region', 'demand_units', 'revenue_usd']].copy()
        
        # Remove duplicates (upsert logic)
        try:
            # Get existing data
            existing = self.db.load_historical_data()
            if not existing.empty:
                # Merge and keep only new records
                existing['date'] = pd.to_datetime(existing['date'])
                merged = df_to_load.merge(
                    existing[['date', 'product_category', 'region']],
                    on=['date', 'product_category', 'region'],
                    how='left',
                    indicator=True
                )
                df_new = merged[merged['_merge'] == 'left_only'][['date', 'product_category', 'region', 'demand_units', 'revenue_usd']]
                
                if not df_new.empty:
                    self.db.save_historical_data(df_new)
                    print(f"Loaded {len(df_new):,} new records")
                else:
                    print("No new records to load")
            else:
                self.db.save_historical_data(df_to_load)
                print(f"Loaded {len(df_to_load):,} records")
        except Exception as e:
            print(f"Error loading data: {e}")
            # Fallback: try direct insert
            self.db.save_historical_data(df_to_load)
        
        return df_to_load
    
    def run_pipeline(self, source_path):
        """Execute full ETL pipeline"""
        print("=" * 60)
        print("Starting ETL Pipeline")
        print("=" * 60)
        
        # Extract
        df_raw = self.extract(source_path)
        print(f"Extracted {len(df_raw):,} raw records")
        
        # Transform
        df_transformed = self.transform(df_raw)
        
        # Load
        df_loaded = self.load(df_transformed)
        
        print("=" * 60)
        print("ETL Pipeline Completed Successfully")
        print("=" * 60)
        
        return df_transformed


if __name__ == '__main__':
    # Example usage
    import sys
    import os
    # Add parent directory to path for imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    pipeline = ETLPipeline()
    
    # Run ETL on generated data
    source_file = 'data/raw/hvac_sales_raw.csv'
    if os.path.exists(source_file):
        pipeline.run_pipeline(source_file)
    else:
        print(f"Source file not found: {source_file}")
        print("Please run data_generator.py first")