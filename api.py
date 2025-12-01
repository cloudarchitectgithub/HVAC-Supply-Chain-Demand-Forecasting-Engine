"""
FastAPI Application for Demand Forecasting
Provides REST API endpoints for forecast predictions, model retraining, and metrics
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
from datetime import datetime, date
import sys
import os

# Add project root to path for imports (api.py is at root level)
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.database import ForecastDatabase
from src.forecast_model import DemandForecastModel, create_baseline_forecast
import pickle

app = FastAPI(
    title="Supply Chain Demand Forecasting API",
    description="API for demand forecasting with ERP integration capabilities",
    version="1.0.0"
)

# CORS middleware for dashboard integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database and model
db = ForecastDatabase()
model = DemandForecastModel()

# Try to load existing model
model_path = 'models/prophet_model.pkl'
if os.path.exists(model_path):
    try:
        model.load_model(model_path)
    except:
        pass


# Pydantic models for request/response
class ForecastRequest(BaseModel):
    product_category: str
    region: str
    periods: Optional[int] = 90


class ForecastResponse(BaseModel):
    product_category: str
    region: str
    forecast_date: date
    predicted_demand: float
    lower_bound: float
    upper_bound: float


class MetricsResponse(BaseModel):
    model_version: str
    metric_name: str
    metric_value: float
    product_category: Optional[str] = None
    region: Optional[str] = None
    evaluation_date: date


class ERPIntegrationRequest(BaseModel):
    """Mock ERP integration payload"""
    product_sku: str
    region_code: str
    current_inventory: int
    lead_time_days: int


@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Supply Chain Demand Forecasting API",
        "version": "1.0.0",
        "endpoints": {
            "/forecast": "Get demand forecast",
            "/forecast/batch": "Get forecasts for multiple products",
            "/retrain": "Retrain forecasting models",
            "/metrics": "Get model performance metrics",
            "/health": "Health check",
            "/erp/integration": "Mock ERP integration endpoint"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "database": "connected",
        "models_loaded": len(model.models),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/forecast", response_model=List[ForecastResponse])
async def get_forecast(request: ForecastRequest):
    """
    Get demand forecast for a specific product-region combination
    """
    try:
        # Generate forecast
        forecast_df = model.predict(
            request.product_category,
            request.region,
            periods=request.periods
        )
        
        # Save to database
        db.save_forecast(forecast_df, model_version=model.model_version)
        
        # Convert to response format
        forecasts = []
        for _, row in forecast_df.iterrows():
            forecasts.append(ForecastResponse(
                product_category=row['product_category'],
                region=row['region'],
                forecast_date=row['forecast_date'].date() if isinstance(row['forecast_date'], pd.Timestamp) else row['forecast_date'],
                predicted_demand=float(row['predicted_demand']),
                lower_bound=float(row['lower_bound']),
                upper_bound=float(row['upper_bound'])
            ))
        
        return forecasts
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating forecast: {str(e)}")


@app.post("/forecast/batch")
async def get_batch_forecast(requests: List[ForecastRequest]):
    """
    Get forecasts for multiple product-region combinations
    """
    results = {}
    
    for request in requests:
        try:
            forecast_df = model.predict(
                request.product_category,
                request.region,
                periods=request.periods
            )
            
            db.save_forecast(forecast_df, model_version=model.model_version)
            
            results[f"{request.product_category}_{request.region}"] = {
                "success": True,
                "forecast_count": len(forecast_df),
                "forecast": forecast_df.to_dict('records')
            }
        except Exception as e:
            results[f"{request.product_category}_{request.region}"] = {
                "success": False,
                "error": str(e)
            }
    
    return results


@app.post("/retrain")
async def retrain_model(background_tasks: BackgroundTasks):
    """
    Retrain forecasting models with latest data
    This would typically be called on a schedule or triggered by new data
    """
    def train_task():
        try:
            # Load latest historical data
            df = db.load_historical_data()
            
            if df.empty:
                return {"error": "No historical data available"}
            
            # Train new model
            global model
            model = DemandForecastModel()
            model.train_all_combinations(df)
            
            # Save model
            model.save_model(model_path)
            
            # Log training
            db.log_training(
                model_version=model.model_version,
                training_samples=len(df),
                model_type="Prophet",
                status="completed"
            )
            
            return {"success": True, "models_trained": len(model.models)}
        except Exception as e:
            return {"error": str(e)}
    
    # Run training in background
    background_tasks.add_task(train_task)
    
    return {
        "message": "Model retraining started in background",
        "status": "processing"
    }


@app.get("/metrics", response_model=List[MetricsResponse])
async def get_metrics(model_version: Optional[str] = None):
    """
    Get model performance metrics
    """
    try:
        metrics_df = db.get_model_metrics(model_version=model_version)
        
        if metrics_df.empty:
            return []
        
        metrics = []
        for _, row in metrics_df.iterrows():
            metrics.append(MetricsResponse(
                model_version=row['model_version'],
                metric_name=row['metric_name'],
                metric_value=float(row['metric_value']),
                product_category=row.get('product_category'),
                region=row.get('region'),
                evaluation_date=pd.to_datetime(row['evaluation_date']).date()
            ))
        
        return metrics
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving metrics: {str(e)}")


@app.post("/erp/integration")
async def erp_integration(request: ERPIntegrationRequest):
    """
    Mock ERP integration endpoint
    In production, this would connect to actual ERP system (e.g., Infor ERP)
    """
    try:
        # Map ERP codes to our product categories and regions
        # In production, this would query ERP system
        product_mapping = {
            "AC-RES-001": "AC_Unit_Residential",
            "AC-COM-001": "AC_Unit_Commercial",
            "HT-RES-001": "Heater_Residential",
            "HT-COM-001": "Heater_Commercial"
        }
        
        region_mapping = {
            "N": "North",
            "S": "South",
            "E": "East",
            "W": "West",
            "C": "Central"
        }
        
        # Get forecast (simplified - would use actual mapping in production)
        product_category = product_mapping.get(request.product_sku, "AC_Unit_Residential")
        region = region_mapping.get(request.region_code, "North")
        
        # Get forecast for lead time period
        forecast_df = model.predict(product_category, region, periods=request.lead_time_days)
        
        # Calculate recommended order quantity
        avg_demand = forecast_df['predicted_demand'].mean()
        safety_stock = avg_demand * 0.2  # 20% safety stock
        recommended_order = max(0, int(avg_demand * request.lead_time_days + safety_stock - request.current_inventory))
        
        return {
            "product_sku": request.product_sku,
            "region_code": request.region_code,
            "current_inventory": request.current_inventory,
            "forecasted_demand_leadtime": float(avg_demand * request.lead_time_days),
            "recommended_order_quantity": recommended_order,
            "reorder_point": int(avg_demand * request.lead_time_days + safety_stock),
            "forecast_details": forecast_df.head(30).to_dict('records')
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ERP integration error: {str(e)}")


@app.get("/products")
async def get_products():
    """Get list of available products and regions"""
    try:
        df = db.load_historical_data()
        
        if df.empty:
            return {"products": [], "regions": []}
        
        products = df['product_category'].unique().tolist()
        regions = df['region'].unique().tolist()
        
        return {
            "products": products,
            "regions": regions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

