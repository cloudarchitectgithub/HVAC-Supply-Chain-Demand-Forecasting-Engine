"""
Streamlit Dashboard for Supply Chain Demand Forecasting
Interactive visualization and analysis tool
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os
import requests

# Add project root to path for imports (dashboard.py is at root level)
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.database import ForecastDatabase
from src.forecast_model import DemandForecastModel

# Page configuration
st.set_page_config(
    page_title="Supply Chain Demand Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
@st.cache_resource
def init_database():
    return ForecastDatabase()

@st.cache_resource
def init_model():
    model = DemandForecastModel()
    model_path = 'models/prophet_model.pkl'
    if os.path.exists(model_path):
        try:
            model.load_model(model_path)
            st.sidebar.success(f"✅ Loaded {len(model.models)} trained models")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Could not load model: {str(e)}")
    else:
        st.sidebar.info("ℹ️ No trained models found. Train models to enable forecasting.")
    return model

db = init_database()
model = init_model()

# API URL (for when API is running)
API_URL = "http://localhost:8000"

# Sidebar
st.sidebar.title("📊 Navigation")
page = st.sidebar.selectbox(
    "Select Page",
    ["Overview", "Historical Data", "Forecasts", "Model Performance", "What-If Analysis", "ERP Integration"]
)

# Helper functions
def load_historical_data(product=None, region=None):
    """Load historical demand data"""
    return db.load_historical_data(product_category=product, region=region)

def get_available_options():
    """Get available products and regions"""
    try:
        df = db.load_historical_data()
        if df.empty:
            return [], []
        products = sorted(df['product_category'].unique().tolist())
        regions = sorted(df['region'].unique().tolist())
        return products, regions
    except Exception as e:
        st.error(f"Error loading options: {str(e)}")
        return [], []

# Main content
if page == "Overview":
    st.title("🏭 Supply Chain Demand Forecasting Engine")
    st.markdown("### AI-Powered Demand Forecasting for HVAC Supply Chain")
    
    # Load data with error handling
    try:
        df = load_historical_data()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please ensure the database is initialized. Run: `python src/data_generator.py` then `python src/etl_pipeline.py`")
        df = pd.DataFrame()
    
    if df.empty:
        st.warning("⚠️ No historical data found. Please run the ETL pipeline first.")
        st.info("**Setup Instructions:**")
        st.code("""
# Step 1: Generate sample data
python src/data_generator.py

# Step 2: Run ETL pipeline
python src/etl_pipeline.py

# Step 3: Refresh this page
        """, language="bash")
    else:

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_demand = df['demand_units'].sum()
        total_revenue = df['revenue_usd'].sum()
        avg_daily_demand = df['demand_units'].mean()
        date_range = f"{df['date'].min().date()} to {df['date'].max().date()}"
        
        with col1:
            st.metric("Total Demand", f"{total_demand:,.0f} units")
        with col2:
            st.metric("Total Revenue", f"${total_revenue:,.0f}")
        with col3:
            st.metric("Avg Daily Demand", f"{avg_daily_demand:,.0f} units")
        with col4:
            st.metric("Date Range", date_range)
        
        # Time series overview
        st.subheader("📈 Demand Trends Over Time")
        
        df_daily = df.groupby('date')['demand_units'].sum().reset_index()
        fig = px.line(df_daily, x='date', y='demand_units', 
                     title="Total Daily Demand",
                     labels={'demand_units': 'Demand (Units)', 'date': 'Date'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Product breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📦 Demand by Product Category")
            product_demand = df.groupby('product_category')['demand_units'].sum().reset_index()
            fig = px.bar(product_demand, x='product_category', y='demand_units',
                        labels={'demand_units': 'Total Demand', 'product_category': 'Product'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🌍 Demand by Region")
            region_demand = df.groupby('region')['demand_units'].sum().reset_index()
            fig = px.pie(region_demand, values='demand_units', names='region',
                        title="Regional Distribution")
            st.plotly_chart(fig, use_container_width=True)

elif page == "Historical Data":
    st.title("📊 Historical Data Analysis")
    
    products, regions = get_available_options()
    
    if not products:
        st.warning("No data available")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            selected_product = st.selectbox("Product Category", ["All"] + products)
        with col2:
            selected_region = st.selectbox("Region", ["All"] + regions)
        
        # Filter data
        product_filter = None if selected_product == "All" else selected_product
        region_filter = None if selected_region == "All" else selected_region
        
        df = load_historical_data(product=product_filter, region=region_filter)
        
        if not df.empty:
            # Display data
            st.subheader("Data Table")
            st.dataframe(df.head(1000), use_container_width=True)
            
            # Time series chart
            st.subheader("Time Series")
            df_ts = df.groupby('date')['demand_units'].sum().reset_index()
            fig = px.line(df_ts, x='date', y='demand_units',
                         title="Historical Demand",
                         labels={'demand_units': 'Demand (Units)', 'date': 'Date'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            st.subheader("Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean", f"{df['demand_units'].mean():.2f}")
            with col2:
                st.metric("Std Dev", f"{df['demand_units'].std():.2f}")
            with col3:
                st.metric("Max", f"{df['demand_units'].max():.0f}")

elif page == "Forecasts":
    st.title("🔮 Demand Forecasts")
    
    products, regions = get_available_options()
    
    if not products:
        st.warning("No data available. Please train models first.")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_product = st.selectbox("Product Category", products)
        with col2:
            selected_region = st.selectbox("Region", regions)
        with col3:
            forecast_periods = st.number_input("Forecast Periods (days)", min_value=7, max_value=365, value=90)
        
        if st.button("Generate Forecast"):
            try:
                # Check if model has trained models
                if not model.models:
                    st.error("⚠️ No trained models found!")
                    st.info("""
                    **Please train models first:**
```bash
                    python src/forecast_model.py
```
                    Then refresh this page.
                    """)
                    st.stop()
                
                # Generate forecast
                forecast_df = model.predict(selected_product, selected_region, periods=forecast_periods)
                
                # Load historical for comparison
                hist_df = load_historical_data(product=selected_product, region=selected_region)
                hist_df = hist_df.sort_values('date').tail(90)  # Last 90 days
                
                # Combine for visualization
                fig = go.Figure()
                
                # Historical data
                if not hist_df.empty:
                    fig.add_trace(go.Scatter(
                        x=hist_df['date'],
                        y=hist_df['demand_units'],
                        mode='lines',
                        name='Historical',
                        line=dict(color='blue', width=2)
                    ))
                
                # Forecast
                fig.add_trace(go.Scatter(
                    x=forecast_df['forecast_date'],
                    y=forecast_df['predicted_demand'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                # Confidence intervals
                fig.add_trace(go.Scatter(
                    x=forecast_df['forecast_date'],
                    y=forecast_df['upper_bound'],
                    mode='lines',
                    name='Upper Bound',
                    line=dict(color='rgba(255,0,0,0.2)', width=1),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_df['forecast_date'],
                    y=forecast_df['lower_bound'],
                    mode='lines',
                    name='Lower Bound',
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.1)',
                    line=dict(color='rgba(255,0,0,0.2)', width=1)
                ))
                
                fig.update_layout(
                    title=f"Demand Forecast: {selected_product} - {selected_region}",
                    xaxis_title="Date",
                    yaxis_title="Demand (Units)",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast summary
                st.subheader("Forecast Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                avg_forecast = forecast_df['predicted_demand'].mean()
                total_forecast = forecast_df['predicted_demand'].sum()
                max_forecast = forecast_df['predicted_demand'].max()
                min_forecast = forecast_df['predicted_demand'].min()
                
                with col1:
                    st.metric("Avg Daily Forecast", f"{avg_forecast:.0f} units")
                with col2:
                    st.metric("Total Forecast", f"{total_forecast:,.0f} units")
                with col3:
                    st.metric("Peak Demand", f"{max_forecast:.0f} units")
                with col4:
                    st.metric("Min Demand", f"{min_forecast:.0f} units")
                
                # Forecast table
                st.subheader("Forecast Details")
                st.dataframe(forecast_df, use_container_width=True)
                
                # Save forecast
                db.save_forecast(forecast_df, model_version=model.model_version)
                st.success("Forecast saved to database")
                
            except Exception as e:
                st.error(f"Error generating forecast: {str(e)}")
                st.info("Make sure the model is trained for this product-region combination")

elif page == "Model Performance":
    st.title("📈 Model Performance Metrics")
    
    # Load metrics
    metrics_df = db.get_model_metrics()
    
    if metrics_df.empty:
        st.warning("No metrics available. Train and evaluate models first.")
    else:
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            model_versions = ["All"] + sorted(metrics_df['model_version'].unique().tolist())
            selected_version = st.selectbox("Model Version", model_versions)
        with col2:
            metric_types = ["All"] + sorted(metrics_df['metric_name'].unique().tolist())
            selected_metric = st.selectbox("Metric Type", metric_types)
        
        # Filter metrics
        filtered_metrics = metrics_df.copy()
        if selected_version != "All":
            filtered_metrics = filtered_metrics[filtered_metrics['model_version'] == selected_version]
        if selected_metric != "All":
            filtered_metrics = filtered_metrics[filtered_metrics['metric_name'] == selected_metric]
        
        # Display metrics
        st.subheader("Metrics Table")
        st.dataframe(filtered_metrics, use_container_width=True)
        
        # Visualizations
        if not filtered_metrics.empty:
            st.subheader("Metrics Visualization")
            
            # Group by metric name
            metric_summary = filtered_metrics.groupby('metric_name')['metric_value'].mean().reset_index()
            fig = px.bar(metric_summary, x='metric_name', y='metric_value',
                        title="Average Metrics by Type",
                        labels={'metric_value': 'Value', 'metric_name': 'Metric'})
            st.plotly_chart(fig, use_container_width=True)

elif page == "What-If Analysis":
    st.title("🎯 What-If Scenario Analysis")
    
    st.markdown("""
    ### Scenario Planning
    Adjust parameters to see how they affect demand forecasts and inventory recommendations.
    """)
    
    products, regions = get_available_options()
    
    if not products:
        st.warning("No data available")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            selected_product = st.selectbox("Product Category", products, key="whatif_product")
            selected_region = st.selectbox("Region", regions, key="whatif_region")
            current_inventory = st.number_input("Current Inventory", min_value=0, value=1000)
        
        with col2:
            lead_time_days = st.number_input("Lead Time (days)", min_value=1, max_value=180, value=30)
            safety_stock_pct = st.slider("Safety Stock %", min_value=0, max_value=50, value=20)
            service_level = st.slider("Target Service Level %", min_value=80, max_value=99, value=95)
        
        if st.button("Run Scenario Analysis"):
            try:
                # Check if model exists
                if not model.models:
                    st.error("⚠️ No trained models available. Please train models first.")
                    st.code("python src/forecast_model.py", language="bash")
                    st.stop()
                
                # Get forecast
                forecast_df = model.predict(selected_product, selected_region, periods=lead_time_days)
                
                # Calculate metrics
                avg_demand = forecast_df['predicted_demand'].mean()
                total_forecasted_demand = forecast_df['predicted_demand'].sum()
                safety_stock = avg_demand * (safety_stock_pct / 100)
                reorder_point = int(avg_demand * lead_time_days + safety_stock)
                recommended_order = max(0, int(reorder_point - current_inventory))
                
                # Display results
                st.subheader("Scenario Results")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Avg Daily Demand", f"{avg_demand:.0f} units")
                with col2:
                    st.metric("Forecasted Demand (Lead Time)", f"{total_forecasted_demand:.0f} units")
                with col3:
                    st.metric("Reorder Point", f"{reorder_point} units")
                with col4:
                    st.metric("Recommended Order", f"{recommended_order} units")
                
                # Inventory status
                st.subheader("Inventory Status")
                if current_inventory < reorder_point:
                    st.warning(f"⚠️ Inventory below reorder point. Order {recommended_order} units.")
                else:
                    st.success(f"✅ Inventory sufficient. No order needed.")
                
                # ROI calculation
                st.subheader("ROI Analysis")
                col1, col2 = st.columns(2)
                
                # Assumptions
                holding_cost_pct = 0.25  # 25% annual holding cost
                unit_cost = 500  # Average unit cost
                stockout_cost = 1000  # Cost per stockout event
                
                # Calculate costs
                excess_inventory = max(0, current_inventory - reorder_point)
                holding_cost = excess_inventory * unit_cost * (holding_cost_pct / 365) * lead_time_days
                
                potential_stockouts = max(0, reorder_point - current_inventory) / avg_demand if avg_demand > 0 else 0
                stockout_cost_risk = potential_stockouts * stockout_cost
                
                with col1:
                    st.metric("Holding Cost (Lead Time)", f"${holding_cost:.2f}")
                with col2:
                    st.metric("Stockout Risk Cost", f"${stockout_cost_risk:.2f}")
                
                # Visualization
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=['Current Inventory', 'Reorder Point', 'Recommended Order'],
                    y=[current_inventory, reorder_point, recommended_order],
                    marker_color=['blue', 'orange', 'green']
                ))
                fig.update_layout(
                    title="Inventory Levels",
                    yaxis_title="Units",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error running scenario: {str(e)}")

elif page == "ERP Integration":
    st.title("🔌 ERP Integration Demo")
    
    st.markdown("""
    ### Mock ERP Integration
    This demonstrates how the forecasting system would integrate with enterprise ERP systems.
    """)
    
    # Mock ERP request form
    with st.form("erp_integration_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            product_sku = st.selectbox(
                "Product SKU",
                ["AC-RES-001", "AC-COM-001", "HT-RES-001", "HT-COM-001"]
            )
            region_code = st.selectbox(
                "Region Code",
                ["N", "S", "E", "W", "C"]
            )
        
        with col2:
            current_inventory = st.number_input("Current Inventory", min_value=0, value=500)
            lead_time_days = st.number_input("Lead Time (days)", min_value=1, max_value=180, value=30)
        
        submitted = st.form_submit_button("Get ERP Recommendation")
        
        if submitted:
            # Check if model is available
            if not model.models:
                st.warning("⚠️ No trained models available. Using default values for demonstration.")
            
            try:
                # Call API (or simulate if API not running)
                try:
                    response = requests.post(
                        f"{API_URL}/erp/integration",
                        json={
                            "product_sku": product_sku,
                            "region_code": region_code,
                            "current_inventory": current_inventory,
                            "lead_time_days": lead_time_days
                        },
                        timeout=5
                    )
                    result = response.json()
                except:
                    # Fallback: simulate API response
                    st.info("API not running, using simulated response")
                    result = {
                        "product_sku": product_sku,
                        "region_code": region_code,
                        "current_inventory": current_inventory,
                        "forecasted_demand_leadtime": 750,
                        "recommended_order_quantity": 250,
                        "reorder_point": 1000
                    }
                
                # Display results
                st.subheader("ERP Integration Response")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Inventory", f"{result['current_inventory']} units")
                with col2:
                    st.metric("Forecasted Demand", f"{result.get('forecasted_demand_leadtime', 0):.0f} units")
                with col3:
                    st.metric("Reorder Point", f"{result.get('reorder_point', 0)} units")
                with col4:
                    st.metric("Recommended Order", f"{result.get('recommended_order_quantity', 0)} units")
                
                # JSON response
                st.subheader("API Response (JSON)")
                st.json(result)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
**Supply Chain Demand Forecasting Engine**

Built for DMI Companies AI Engineer position demonstration.

Features:
- ETL Pipeline
- Prophet Forecasting
- FastAPI Integration
- ERP Integration
- ROI Analysis
""")

