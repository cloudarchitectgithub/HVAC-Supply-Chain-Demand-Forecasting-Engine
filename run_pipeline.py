#!/usr/bin/env python3
"""
Main script to run the complete ETL and model training pipeline
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_generator import generate_hvac_sales_data
from src.etl_pipeline import ETLPipeline
from src.forecast_model import DemandForecastModel
from src.database import ForecastDatabase


def main():
    print("=" * 70)
    print("Supply Chain Demand Forecasting Engine - Pipeline Runner")
    print("=" * 70)
    
    # Step 1: Generate sample data (if not exists)
    raw_data_path = 'data/raw/hvac_sales_raw.csv'
    if not os.path.exists(raw_data_path):
        print("\n[Step 1/4] Generating sample HVAC sales data...")
        generate_hvac_sales_data()
        print("✅ Sample data generated")
    else:
        print("\n[Step 1/4] Sample data already exists, skipping generation...")
    
    # Step 2: Run ETL Pipeline
    print("\n[Step 2/4] Running ETL Pipeline...")
    pipeline = ETLPipeline()
    try:
        df_transformed = pipeline.run_pipeline(raw_data_path)
        print("✅ ETL pipeline completed")
    except Exception as e:
        print(f"❌ ETL pipeline failed: {str(e)}")
        return
    
    # Step 3: Train Forecasting Models
    print("\n[Step 3/4] Training Forecasting Models...")
    db = ForecastDatabase()
    df_historical = db.load_historical_data()
    
    if df_historical.empty:
        print("❌ Error: No historical data found after ETL")
        return
    
    print(f"Loaded {len(df_historical):,} historical records")
    
    model = DemandForecastModel(forecast_periods=90)
    trained_count = model.train_all_combinations(df_historical)
    
    if trained_count == 0:
        print("⚠️ Warning: No models were trained")
        return
    
    print(f"✅ Trained {trained_count} models")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model.save_model('models/prophet_model.pkl')
    print("✅ Model saved to models/prophet_model.pkl")
    
    # Step 4: Evaluate Models
    print("\n[Step 4/4] Evaluating Models...")
    products = df_historical['product_category'].unique()[:3]  # Evaluate first 3
    regions = df_historical['region'].unique()[:2]  # Evaluate first 2
    
    evaluated_count = 0
    for product in products:
        for region in regions:
            try:
                metrics = model.evaluate(df_historical, product, region)
                if metrics:
                    db.save_model_metrics(metrics, model.model_version, product, region)
                    print(f"  {product} - {region}: MAPE={metrics['MAPE']:.2f}%, RMSE={metrics['RMSE']:.2f}")
                    evaluated_count += 1
            except Exception as e:
                print(f"  ⚠️ Could not evaluate {product} - {region}: {str(e)}")
                pass
    
    print(f"\n✅ Evaluated {evaluated_count} model combinations")
    
    # Log training
    db.log_training(
        model_version=model.model_version,
        training_samples=len(df_historical),
        model_type="Prophet",
        status="completed",
        notes=f"Trained {trained_count} models, evaluated {evaluated_count}"
    )
    
    print("\n" + "=" * 70)
    print("✅ Pipeline Completed Successfully!")
    print("=" * 70)
    print(f"\n📊 Results Summary:")
    print(f"  • Historical records: {len(df_historical):,}")
    print(f"  • Models trained: {trained_count}")
    print(f"  • Models evaluated: {evaluated_count}")
    print(f"\n🚀 Next steps:")
    print(f"  1. Start Dashboard: streamlit run dashboard.py")
    print(f"  2. Start API: python api.py")
    print(f"  3. View database: data/forecast_db.sqlite")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)