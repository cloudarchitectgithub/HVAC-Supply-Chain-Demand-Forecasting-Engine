"""
Data Generator for HVAC Sales Data
Creates realistic time series data for demand forecasting demonstration
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def generate_hvac_sales_data(start_date='2020-01-01', end_date='2024-12-31', output_path='data/raw'):
    """
    Generate synthetic HVAC sales data with realistic patterns:
    - Seasonal trends (higher in summer/winter)
    - Product categories (AC units, heaters, parts)
    - Regional variations
    - Random noise
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Product categories
    products = ['AC_Unit_Residential', 'AC_Unit_Commercial', 'Heater_Residential', 
                'Heater_Commercial', 'HVAC_Parts', 'Maintenance_Kit']
    
    # Regions
    regions = ['North', 'South', 'East', 'West', 'Central']
    
    # Generate base demand with seasonality
    data = []
    
    for date in dates:
        # Seasonal factor (higher in summer and winter)
        day_of_year = date.timetuple().tm_yday
        seasonal_factor = 1 + 0.4 * np.sin(2 * np.pi * (day_of_year - 80) / 365)  # Summer peak
        seasonal_factor += 0.3 * np.sin(2 * np.pi * (day_of_year - 350) / 365)  # Winter peak
        
        # Weekly pattern (lower on weekends)
        weekday_factor = 0.7 if date.weekday() >= 5 else 1.0
        
        # Yearly growth trend
        years_elapsed = (date - pd.to_datetime(start_date)).days / 365.25
        growth_factor = 1 + 0.08 * years_elapsed  # 8% annual growth
        
        for product in products:
            for region in regions:
                # Base demand varies by product and region
                base_demand = {
                    'AC_Unit_Residential': 50,
                    'AC_Unit_Commercial': 20,
                    'Heater_Residential': 40,
                    'Heater_Commercial': 15,
                    'HVAC_Parts': 200,
                    'Maintenance_Kit': 100
                }[product]
                
                # Regional adjustments
                region_multiplier = {
                    'North': 1.2 if 'Heater' in product else 0.8,
                    'South': 1.3 if 'AC' in product else 0.7,
                    'East': 1.0,
                    'West': 1.1,
                    'Central': 0.9
                }[region]
                
                # Calculate demand with all factors
                demand = base_demand * seasonal_factor * weekday_factor * growth_factor * region_multiplier
                
                # Add random noise
                demand = max(0, np.random.normal(demand, demand * 0.15))
                
                data.append({
                    'date': date,
                    'product_category': product,
                    'region': region,
                    'demand_units': int(demand),
                    'revenue_usd': int(demand * np.random.uniform(200, 2000))
                })
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_file = os.path.join(output_path, 'hvac_sales_raw.csv')
    df.to_csv(output_file, index=False)
    print(f"Generated {len(df):,} records saved to {output_file}")
    
    return df


if __name__ == '__main__':
    print("Generating HVAC sales data...")
    df = generate_hvac_sales_data()
    print(f"\nData Summary:")
    print(df.groupby('product_category')['demand_units'].sum())
    print(f"\nTotal records: {len(df):,}")

