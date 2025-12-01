# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run the Complete Pipeline

```bash
python run_pipeline.py
```

This will:
- ✅ Generate sample HVAC sales data (5 years)
- ✅ Run ETL pipeline with feature engineering
- ✅ Train Prophet forecasting models
- ✅ Evaluate model performance

**Expected output:** ~2-5 minutes depending on your machine

### Step 3: Start the API

In a new terminal:

```bash
python -m uvicorn src.api:app --reload
```

API will be available at: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### Step 4: Start the Dashboard

In another terminal:

```bash
streamlit run src/dashboard.py
```

Dashboard opens automatically at: `http://localhost:8501`

---

## 📊 Test the System

### Test API Endpoint

```bash
curl -X POST "http://localhost:8000/forecast" \
  -H "Content-Type: application/json" \
  -d '{
    "product_category": "AC_Unit_Residential",
    "region": "North",
    "periods": 30
  }'
```

### Test ERP Integration

```bash
curl -X POST "http://localhost:8000/erp/integration" \
  -H "Content-Type: application/json" \
  -d '{
    "product_sku": "AC-RES-001",
    "region_code": "N",
    "current_inventory": 500,
    "lead_time_days": 30
  }'
```

---

## 🎯 Dashboard Features

1. **Overview**: Key metrics and trends
2. **Historical Data**: Explore past demand
3. **Forecasts**: Generate predictions
4. **Model Performance**: View accuracy metrics
5. **What-If Analysis**: Scenario planning with ROI
6. **ERP Integration**: Test integration endpoints

---

## 🔧 Troubleshooting

### Issue: Import errors

**Solution:** Make sure you're in the project root directory:
```bash
cd "Supply Chain Demand Forecasting Engine"
```

### Issue: Prophet installation fails

**Solution:** Install Prophet dependencies:
```bash
pip install prophet
# On Mac/Linux, you may need:
# conda install -c conda-forge prophet
```

### Issue: No data in dashboard

**Solution:** Run the pipeline first:
```bash
python run_pipeline.py
```

### Issue: Models not trained

**Solution:** Check that you have at least 30 days of data per product region combination.

---

## 📁 Project Structure

```
Supply Chain Demand Forecasting Engine/
├── src/              # Source code
├── data/             # Data files (generated)
├── models/           # Trained models (generated)
├── run_pipeline.py   # Main pipeline runner
└── requirements.txt  # Dependencies
```

---

## ✅ Verification Checklist

- [ ] Dependencies installed
- [ ] Pipeline runs successfully
- [ ] Database created (`data/forecast_db.sqlite`)
- [ ] Models trained (`models/prophet_model.pkl`)
- [ ] API starts without errors
- [ ] Dashboard loads and displays data
- [ ] Can generate forecasts via API
- [ ] Can view forecasts in dashboard

---

## 🎓 Next Steps

1. Explore the dashboard features
2. Try different product region combinations
3. Run what if scenarios
4. Review the code structure
5. Check the README.md for detailed documentation

---



