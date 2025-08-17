# 🚚 MilesAhead - Smart Delivery ETA & Route Optimization

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28%2B-red)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/xgboost-ML%20Model-green)](https://xgboost.readthedocs.io/)
[![OSRM](https://img.shields.io/badge/OSRM-Route%20API-orange)](http://project-osrm.org/)

**MilesAhead** is an intelligent delivery time prediction and route optimization platform that uses machine learning and real-world data to provide accurate delivery ETAs and interactive route visualizations.

## 🌟 **Key Features**

### 🔮 **Smart Prediction Engine**
- **Machine Learning Model**: XGBoost-based prediction with **84.2% R² accuracy**
- **Real Road Distances**: Integration with OSRM API for actual driving routes
- **Vehicle Constraints**: Speed and distance limits for different vehicle types
- **Multi-factor Analysis**: Weather, traffic, agent ratings, and more

### 🗺️ **Interactive Maps**
- **Real-time Route Visualization**: Dynamic maps with actual driving paths
- **Multiple Route Support**: Compare different delivery routes
- **Interactive Markers**: Store and delivery location details
- **Distance & Time Display**: Real distance and estimated delivery time

### 📊 **Business Intelligence Dashboard**
- **Exploratory Data Analysis (EDA)**: Comprehensive data insights
- **Performance Metrics**: Delivery time trends and patterns
- **Agent Analytics**: Rating distributions and performance analysis
- **Route Optimization**: Efficiency recommendations

### 🚀 **Multi-Page Web Application**
- **User-Friendly Interface**: Built with Streamlit
- **Real-time Predictions**: Instant delivery time estimates
- **Map Integration**: Folium-powered interactive maps
- **Data Visualization**: Charts, graphs, and analytics

## 🛠️ **Technology Stack**

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Machine Learning** | XGBoost, Scikit-learn | Delivery time prediction |
| **Web Framework** | Streamlit | Interactive web application |
| **Mapping** | Folium, OSRM API | Route visualization and distance calculation |
| **Data Processing** | Pandas, NumPy | Data manipulation and analysis |
| **Visualization** | Plotly, Matplotlib, Seaborn | Charts and graphs |
| **Backend** | Python 3.8+ | Core application logic |

## 📁 **Project Structure**

```
MilesAhead/
├── 📊 streamlit_app.py                 # Main Streamlit application
├── 📋 requirements.txt                 # Python dependencies
├── 🔧 run_app.sh                      # Application launcher script
├── 📖 README.md                       # Project documentation
│
├── 🧠 src/                            # Source code
│   ├── 🎯 model_training/
│   │   └── predict_evaluate.py        # ML prediction engine
│   ├── 🗺️ map_components/
│   │   └── map_visualisation.py       # Map and route visualization
│   ├── ⚙️ feature_engineering/        # Data preprocessing
│   │   ├── encoding.py
│   │   ├── feature_extraction.py
│   │   ├── feature_selection.py
│   │   ├── outlier.py
│   │   └── scaling.py
│   └── 📊 data_processing/            # Data handling utilities
│
├── 🤖 models/                         # ML models (excluded from git)
│   ├── Best_Tuned_XGBoost_Model.joblib
│   └── preprocessing_pipeline.joblib
│
├── 📁 data/                           # Datasets (excluded from git)
│   └── processed/
│
├── 📊 outputs/                        # Generated outputs
└── 📚 notebooks/                      # Jupyter notebooks for analysis
```

## 🚀 **Quick Start**

### **Prerequisites**
- Python 3.8 or higher
- pip package manager
- Internet connection (for OSRM API)

### **Installation**

1. **Clone the repository**
   ```bash
   git clone https://github.com/anuragchaubey1224/MilesAhead.git
   cd MilesAhead
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   # Option 1: Direct Streamlit
   streamlit run streamlit_app.py
   
   # Option 2: Using launcher script
   chmod +x run_app.sh
   ./run_app.sh
   ```

5. **Open in browser**
   - Navigate to `http://localhost:8501`
   - Start predicting delivery times!

## 📖 **How to Use**

### **🎯 Delivery Prediction**

1. **Input Delivery Details**
   - **Store Location**: Enter store coordinates or select from map
   - **Delivery Location**: Specify drop-off coordinates
   - **Vehicle Type**: Choose from motorcycle, scooter, bicycle, electric scooter
   - **Agent Details**: Age and rating
   - **Conditions**: Weather and traffic status

2. **Get Predictions**
   - **Estimated Time**: ML-powered delivery time prediction
   - **Real Distance**: Actual driving distance via OSRM API
   - **Effective Speed**: Calculated based on vehicle constraints
   - **Route Map**: Interactive visualization of the delivery route

3. **View Results**
   - **Map Visualization**: See the actual route on an interactive map
   - **Delivery Details**: Comprehensive breakdown of time and distance
   - **Performance Metrics**: Speed, efficiency, and constraint validation

### **📊 Business Insights**

1. **Exploratory Data Analysis**
   - Delivery time distribution patterns
   - Geographic heat maps
   - Vehicle performance comparisons
   - Seasonal and temporal trends

2. **Agent Performance**
   - Rating distributions
   - Delivery time correlations
   - Performance benchmarking
   - Improvement recommendations

3. **Route Optimization**
   - Distance efficiency analysis
   - Time vs. distance correlations
   - Vehicle suitability recommendations
   - Cost optimization insights

## 🔧 **Core Components**

### **1. ML Prediction Engine** (`predict_evaluate.py`)

```python
# Example usage
from model_training.predict_evaluate import get_prediction

delivery_data = {
    'store_latitude': 19.0760,
    'store_longitude': 72.8777,
    'drop_latitude': 19.0825,
    'drop_longitude': 72.8231,
    'vehicle': 'motorcycle',
    'agent_rating': 4.5,
    'agent_age': 28,
    'traffic': 'Medium',
    'weather': 'Sunny'
}

predicted_time, map_data = get_prediction(delivery_data)
print(f"Estimated delivery time: {predicted_time:.1f} minutes")
```

**Features:**
- ✅ XGBoost model with 84.2% accuracy
- ✅ OSRM API integration for real distances
- ✅ Vehicle constraint validation
- ✅ Fallback mechanisms for reliability
- ✅ Comprehensive error handling

### **2. Map Visualization** (`map_visualisation.py`)

```python
# Example usage
from map_components.map_visualisation import visualize_delivery_routes_on_map

# Visualize delivery routes
map_html = visualize_delivery_routes_on_map(delivery_data)
```

**Features:**
- ✅ Real-time route plotting via OSRM
- ✅ Interactive markers and popups
- ✅ Distance and time calculations
- ✅ Multiple route support
- ✅ Custom styling and themes

### **3. Streamlit Application** (`streamlit_app.py`)

**Multi-page application with:**
- 🏠 **Home**: Welcome and overview
- 🎯 **Prediction**: Delivery time estimation
- 📊 **Analytics**: EDA and business insights
- 🗺️ **Maps**: Interactive route visualization
- 📈 **Dashboard**: Performance metrics

## 🎯 **Vehicle Constraints**

| Vehicle Type | Speed Range | Max Distance | Use Case |
|--------------|-------------|--------------|----------|
| 🏍️ **Motorcycle** | 25-60 km/h | 50 km | Long-distance, fast delivery |
| 🛵 **Scooter** | 20-45 km/h | 30 km | Medium-distance, urban delivery |
| 🚲 **Bicycle** | 10-20 km/h | 15 km | Short-distance, eco-friendly |
| ⚡ **Electric Scooter** | 15-35 km/h | 25 km | Urban, environmentally conscious |

## 📊 **API Integration**

### **OSRM (Open Source Routing Machine)**

The application integrates with OSRM API for real-world routing:

```python
# Automatic real distance calculation
distance_data = get_real_route_from_osrm(
    pickup_lon=72.8777, pickup_lat=19.0760,
    drop_lon=72.8231, drop_lat=19.0825
)

# Returns: distance_km, duration_minutes, route_geometry
```

**Benefits:**
- ✅ Real road distances (not straight-line)
- ✅ Actual driving routes
- ✅ Traffic-aware routing
- ✅ Detailed route geometry
- ✅ Fallback to Haversine if API unavailable

## 🔍 **Machine Learning Model**

### **Model Performance**
- **Algorithm**: XGBoost Regressor
- **Accuracy**: 84.2% R² Score
- **Features**: 15+ engineered features
- **Training Data**: Real delivery dataset
- **Validation**: Cross-validation with temporal splits

### **Feature Engineering**
- **Temporal Features**: Hour, day of week, seasonality
- **Geospatial Features**: Distance, zone mapping
- **Agent Features**: Age, rating, experience
- **Contextual Features**: Weather, traffic, category
- **Derived Features**: Speed ratios, efficiency metrics

### **Model Pipeline**
1. **Data Preprocessing**: Outlier removal, feature scaling
2. **Feature Selection**: Tree-based feature importance
3. **Model Training**: Hyperparameter-tuned XGBoost
4. **Validation**: Time-series cross-validation
5. **Deployment**: Joblib serialization for production

## 📈 **Performance Metrics**

### **Prediction Accuracy**
- **R² Score**: 84.2%
- **Mean Absolute Error**: ~5.2 minutes
- **Root Mean Square Error**: ~8.1 minutes
- **Prediction Speed**: <100ms per prediction

### **System Performance**
- **API Response Time**: <2 seconds for OSRM calls
- **Map Rendering**: <3 seconds for route visualization
- **Application Load Time**: <5 seconds initial load
- **Memory Usage**: ~150MB for complete application

## 🔧 **Configuration**

### **Environment Variables**

Create a `.env` file for configuration:

```bash
# Optional API configurations
OSRM_BASE_URL=http://router.project-osrm.org
MAP_STYLE=OpenStreetMap
DEBUG_MODE=False

# Model paths (auto-detected)
MODEL_DIR=./models
DATA_DIR=./data
OUTPUT_DIR=./outputs
```

### **Model Requirements**

The application expects these model files in the `models/` directory:
- `Best_Tuned_XGBoost_Model.joblib` - Main prediction model
- `preprocessing_pipeline.joblib` - Data preprocessing pipeline

*Note: Model files are excluded from git due to size. Contact maintainer for access.*

## 🚀 **Deployment**

### **Local Development**
```bash
streamlit run streamlit_app.py --server.headless=true --server.port=8501
```

### **Production Deployment**

**Docker Deployment:**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.headless=true", "--server.port=8501"]
```

**Cloud Platforms:**
- **Streamlit Cloud**: Direct GitHub integration
- **Heroku**: Web app deployment
- **AWS/GCP**: Container deployment
- **Railway**: Simple deployment platform

## 🤝 **Contributing**

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Add tests** (if applicable)
5. **Commit your changes**
   ```bash
   git commit -m "Add amazing feature"
   ```
6. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
7. **Open a Pull Request**

### **Development Guidelines**
- Follow PEP 8 Python style guidelines
- Add docstrings to new functions
- Include error handling for external API calls
- Test with sample data before submitting
- Update documentation for new features

## 📋 **To-Do / Future Enhancements**

### **Short Term**
- [ ] Add authentication and user management
- [ ] Implement delivery history tracking
- [ ] Add more vehicle types and constraints
- [ ] Enhanced error reporting and logging
- [ ] Mobile-responsive design improvements

### **Medium Term**
- [ ] Real-time tracking integration
- [ ] Advanced route optimization algorithms
- [ ] Machine learning model retraining pipeline
- [ ] API rate limiting and caching
- [ ] Multi-language support

### **Long Term**
- [ ] Integration with delivery service APIs
- [ ] Advanced analytics and reporting
- [ ] Mobile application development
- [ ] Real-time traffic integration
- [ ] Predictive demand modeling

## 🐛 **Troubleshooting**

### **Common Issues**

**1. OSRM API Errors**
```
Error: Failed to connect to OSRM server
Solution: Check internet connection, API may be temporarily down
Fallback: Application uses Haversine distance calculation
```

**2. Model Loading Issues**
```
Error: Model file not found
Solution: Ensure model files are in the models/ directory
Contact: Reach out for model file access
```

**3. Streamlit Performance**
```
Issue: Slow application loading
Solution: Clear browser cache, restart application
Optimization: Close other browser tabs to free memory
```

**4. Map Rendering Problems**
```
Issue: Maps not displaying
Solution: Check browser JavaScript, try different browser
Alternative: Use fallback coordinate display
```

## 📞 **Support & Contact**

### **Getting Help**
- 📧 **Email**: anuragchaubey1224@gmail.com
- 💬 **GitHub Issues**: [Create an issue](https://github.com/anuragchaubey1224/MilesAhead/issues)
- 📖 **Documentation**: Check this README and code comments
- 🔍 **Stack Overflow**: Tag questions with `milesahead-delivery`

### **Reporting Bugs**
When reporting bugs, please include:
- Operating system and Python version
- Error messages and stack traces
- Steps to reproduce the issue
- Sample input data (if applicable)
- Browser and version (for web issues)

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### **MIT License Summary**
- ✅ Commercial use allowed
- ✅ Modification allowed
- ✅ Distribution allowed
- ✅ Private use allowed
- ❗ License and copyright notice required
- ❗ No warranty provided

## 🙏 **Acknowledgments**

### **Technologies & Libraries**
- **XGBoost**: Gradient boosting framework
- **Streamlit**: Web application framework
- **OSRM**: Open source routing engine
- **Folium**: Interactive mapping library
- **Scikit-learn**: Machine learning toolkit
- **Plotly**: Interactive visualization library

### **Data Sources**
- Delivery dataset for model training
- OpenStreetMap for mapping data
- Weather and traffic APIs for contextual data

### **Special Thanks**
- Open source community for tools and libraries
- OSRM project for routing services
- Streamlit team for the amazing framework
- Contributors and testers

---

## 📊 **Quick Stats**

| Metric | Value |
|--------|-------|
| **Lines of Code** | ~2,500+ |
| **Python Files** | 15+ |
| **ML Accuracy** | 84.2% R² |
| **API Integration** | OSRM Routing |
| **Visualization** | Interactive Maps |
| **Deployment** | Streamlit Cloud Ready |

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

**🚚 Built with ❤️ for smarter delivery logistics**

[🔗 **Live Demo**](https://milesahead-delivery.streamlit.app) | [📖 **Documentation**](https://github.com/anuragchaubey1224/MilesAhead) | [🐛 **Report Bug**](https://github.com/anuragchaubey1224/MilesAhead/issues)

</div>

---

*Last updated: August 17, 2025 | Version: 2.0*
