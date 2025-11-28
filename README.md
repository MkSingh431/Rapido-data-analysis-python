# 🚖 Rapido Ride Analytics Dashboard

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/MkSingh431/Rapido/graphs/commit-activity)

A production-ready, interactive Streamlit dashboard for comprehensive Rapido ride-sharing analytics with advanced ML forecasting, real-time insights, and enterprise-grade visualizations.

## 🎯 Key Features

### 📊 Advanced Analytics
- **Multi-dimensional Filtering**: Dynamic filters with cross-validation
- **Real-time KPIs**: Revenue, rides, conversion rates, and operational metrics
- **Geospatial Analysis**: Route optimization and demand heatmaps
- **Cohort Analysis**: Customer retention and lifetime value tracking

### 📈 Machine Learning & Forecasting
- **Time Series Models**: Holt-Winters, ARIMA, and Facebook Prophet
- **Demand Prediction**: Peak hour and seasonal forecasting
- **Anomaly Detection**: Automated outlier identification
- **A/B Testing Framework**: Statistical significance testing

### 🎨 Interactive Visualizations
- **Dynamic Dashboards**: Plotly-powered interactive charts
- **Sankey Flow Diagrams**: Route and payment flow analysis
- **Heatmaps**: Temporal and geographical demand patterns
- **Statistical Distributions**: Distance, duration, and fare analysis

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- 4GB+ RAM recommended
- Modern web browser

### Installation

```bash
# Clone repository
git clone https://github.com/MkSingh431/Rapido.git
cd Rapido

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
# Create config file
cp config.example.yaml config.yaml

# Edit configuration
vim config.yaml
```

### Launch Dashboard

```bash
# Development mode
streamlit run rapido_app.py --server.port 8501

# Production deployment
streamlit run rapido_app.py --server.headless true --server.port 80
```

## 📁 Project Structure

```
Rapido/
├── rapido_app.py           # Main Streamlit application
├── data/
│   ├── rides_data.csv      # Sample dataset
│   └── schema.json         # Data validation schema
├── models/
│   ├── forecasting.py      # ML forecasting models
│   └── analytics.py        # Statistical analysis
├── utils/
│   ├── data_loader.py      # Data processing utilities
│   ├── visualizations.py   # Custom chart components
│   └── validators.py       # Data validation
├── tests/
│   ├── test_models.py      # Unit tests
│   └── test_data.py        # Data quality tests
├── requirements.txt        # Python dependencies
├── config.yaml            # Application configuration
├── Dockerfile             # Container deployment
└── README.md              # Documentation
```

## 📊 Data Schema

### Required Columns

| Column | Type | Description | Example |
|--------|------|-------------|----------|
| `date` | datetime | Ride timestamp | 2024-01-15 14:30:00 |
| `ride_id` | string | Unique identifier | RID_001234 |
| `ride_status` | enum | completed/cancelled | completed |
| `services` | string | Service type | bike/auto/cab |
| `payment_method` | enum | cash/upi/card | upi |
| `total_fare` | float | Ride cost (₹) | 125.50 |
| `distance` | float | Distance (km) | 8.5 |
| `duration` | float | Duration (hours) | 0.75 |
| `time` | time | Ride start time | 14:30:00 |
| `source` | string | Pickup location | Koramangala |
| `destination` | string | Drop location | Whitefield |

### Optional Enhancements
- `customer_id`: Customer segmentation
- `driver_id`: Driver performance analysis
- `weather`: Weather impact analysis
- `surge_multiplier`: Dynamic pricing analysis

## 🔧 Advanced Configuration

### Environment Variables

```bash
# .env file
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=false
DATA_REFRESH_INTERVAL=300
CACHE_TTL=3600
LOG_LEVEL=INFO
```

### Performance Optimization

```python
# config.yaml
performance:
  cache_size: 1000
  chunk_size: 10000
  parallel_processing: true
  memory_limit: "4GB"
```

## 🚀 Deployment Options

### Docker Deployment

```bash
# Build image
docker build -t rapido-dashboard .

# Run container
docker run -p 8501:8501 rapido-dashboard
```

### Cloud Deployment

```bash
# Streamlit Cloud
streamlit deploy

# Heroku
git push heroku main

# AWS EC2
# See deployment/aws-setup.md
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Coverage report
pytest --cov=. tests/

# Performance tests
pytest tests/test_performance.py --benchmark-only
```

## 📈 Performance Metrics

- **Load Time**: < 3 seconds for 100K records
- **Memory Usage**: < 2GB for standard datasets
- **Concurrent Users**: Supports 50+ simultaneous users
- **Data Processing**: 1M+ records in < 30 seconds

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Pre-commit hooks
pre-commit install

# Code formatting
black rapido_app.py
flake8 rapido_app.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Acknowledgments

- **Streamlit Team** for the amazing framework
- **Plotly** for interactive visualizations
- **Facebook Prophet** for time series forecasting
- **Open Source Community** for continuous inspiration

## 📞 Support & Contact

**Motilal Das**  
🎯 Data Science | Business Intelligence | ML Engineering  
📧 [mks465261@gmail.com](mailto:mks465261@gmail.com)  
💼 [LinkedIn](https://www.linkedin.com/in/motilal-das-42b4a9254/)  
🐙 [GitHub](https://github.com/MkSingh431/MkSingh431)  
🏆 [HackerRank](https://www.hackerrank.com/mk_singh431)  
📊 [Tableau Portfolio](https://public.tableau.com/profile/motilal.das)

---

⭐ **Star this repository if it helped you!** ⭐