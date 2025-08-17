# 🚚 MilesAhead - Smart Delivery ETA & Route Optimization
# advanced Multi-Page Streamlit Application

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import os
import sys
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# path setups to import local modules
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

# import custom modules
from model_training.predict_evaluate import get_prediction
from map_components.map_visualisation import visualize_delivery_routes_on_map

# --- Page Configuration ---
st.set_page_config(
    page_title="MilesAhead - ML Based Delivery Analytics",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# MilesAhead\nSmart Delivery ETA & Route Optimization Platform"
    }
)

# custom css for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #2d3436, #636e72);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 4px solid #667eea;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.4);
    }
    
    .eda-section {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #4a5568;
        color: #e2e8f0;
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    .stSelectbox > div > div {
        border-radius: 10px;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# data loading functions
@st.cache_data
def load_data():
    """Load the processed delivery dataset"""
    try:
        df = pd.read_csv("data/processed/delivery_data_final.csv")
        # Convert datetime columns
        df['Order_Date'] = pd.to_datetime(df['Order_Date'])
        return df
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'data/processed/delivery_data_final.csv' exists.")
        return None

@st.cache_data
def get_data_summary(df):
    """Get comprehensive data summary"""
    summary = {
        'total_orders': len(df),
        'avg_delivery_time': df['Delivery_Time'].mean(),
        'unique_agents': df['Agent_Age'].nunique(),
        'unique_categories': df['Category'].nunique(),
        'date_range': (df['Order_Date'].min(), df['Order_Date'].max()),
        'avg_distance': df['real_distance_km'].mean() if 'real_distance_km' in df.columns else None
    }
    return summary

#  sidebar navigation
def render_sidebar():
    """Render the sidebar navigation"""
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h1 style="color: #667eea;">🚚 MilesAhead</h1>
        <p style="color: #666;">Smart Delivery Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["🏠 Home", "📊 EDA Analysis", "🎯 Delivery Prediction", "📈 Business Insights", "ℹ️ About"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Stats")
    
    # load data for quick stats
    df = load_data()
    if df is not None:
        summary = get_data_summary(df)
        st.sidebar.metric("Total Orders", f"{summary['total_orders']:,}")
        st.sidebar.metric("Avg Delivery Time", f"{summary['avg_delivery_time']:.1f} min")
        if summary['avg_distance']:
            st.sidebar.metric("Avg Distance", f"{summary['avg_distance']:.1f} km")
    
    return page

# --- Page: Home ---
def render_home_page():
    """Render the home page"""
    st.markdown("""
    <div class="main-header">
        <h1>🚚 Welcome to MilesAhead</h1>
        <h3>Smart Delivery ETA & Route Optimization Platform</h3>
        <p>Predict delivery times with AI and visualize optimal routes in real-time</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Data Analytics</h3>
            <p>Comprehensive EDA analysis of delivery patterns, trends, and insights from historical data.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 AI Predictions</h3>
            <p>Machine learning-powered delivery time predictions using real-world factors and OSRM routing.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🗺️ Route Visualization</h3>
            <p>Interactive maps showing real road routes, optimized paths, and delivery analytics.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Dataset Overview
    df = load_data()
    if df is not None:
        st.markdown("## 📈 Dataset Overview")
        
        summary = get_data_summary(df)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Orders", f"{summary['total_orders']:,}")
        with col2:
            st.metric("Average Delivery Time", f"{summary['avg_delivery_time']:.1f} min")
        with col3:
            st.metric("Unique Categories", summary['unique_categories'])
        with col4:
            if summary['avg_distance']:
                st.metric("Average Distance", f"{summary['avg_distance']:.1f} km")
        
        # Quick visualizations
        st.markdown("### 📊 Quick Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Delivery time distribution
            fig_dist = px.histogram(
                df, x='Delivery_Time', 
                title="Delivery Time Distribution",
                nbins=50,
                color_discrete_sequence=['#667eea']
            )
            fig_dist.update_layout(
                showlegend=False, 
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                title_font_color='white'
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            # Category breakdown
            category_counts = df['Category'].value_counts().head(10)
            fig_cat = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Top Order Categories"
            )
            fig_cat.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                title_font_color='white'
            )
            st.plotly_chart(fig_cat, use_container_width=True)

# --- Page: EDA Analysis ---
def render_eda_page():
    """Render comprehensive EDA analysis page"""
    st.markdown("# 📊 Exploratory Data Analysis")
    
    df = load_data()
    if df is None:
        return
    
    # Data Overview Section
    st.markdown("## 🔍 Data Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Dataset Information")
        st.write(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
        st.write(f"**Date Range:** {df['Order_Date'].min().date()} to {df['Order_Date'].max().date()}")
        st.write(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    with col2:
        st.markdown("### Missing Values")
        missing_data = df.isnull().sum()
        if missing_data.sum() == 0:
            st.success("✅ No missing values!")
        else:
            st.write(missing_data[missing_data > 0])
    
    # show sample data
    with st.expander("👁️ View Sample Data"):
        st.dataframe(df.head(10), use_container_width=True)
    
    # target variable analysis
    st.markdown("## 🎯 Target Variable Analysis: Delivery Time")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # statistical summary
        st.markdown("### 📈 Statistical Summary")
        delivery_stats = df['Delivery_Time'].describe()
        for stat, value in delivery_stats.items():
            st.write(f"**{stat.title()}:** {value:.2f} minutes")
    
    with col2:
        # distribution plot
        fig_delivery = go.Figure()
        fig_delivery.add_trace(go.Histogram(
            x=df['Delivery_Time'],
            nbinsx=50,
            name="Delivery Time",
            marker_color='#667eea'
        ))
        fig_delivery.update_layout(
            title="Delivery Time Distribution",
            xaxis_title="Delivery Time (minutes)",
            yaxis_title="Frequency",
            height=400
        )
        st.plotly_chart(fig_delivery, use_container_width=True)
    
    # categorical Variables Analysis
    st.markdown("## 📊 Categorical Variables Analysis")
    
    categorical_cols = ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category']
    
    tabs = st.tabs([col.replace('_', ' ').title() for col in categorical_cols])
    
    for i, col in enumerate(categorical_cols):
        with tabs[i]:
            col1, col2 = st.columns(2)
            
            with col1:
                # Value counts
                value_counts = df[col].value_counts()
                st.markdown(f"### {col} Distribution")
                for val, count in value_counts.items():
                    percentage = (count / len(df)) * 100
                    st.write(f"**{val}:** {count:,} ({percentage:.1f}%)")
            
            with col2:
                # Bar chart
                fig_cat = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"{col} Distribution",
                    color=value_counts.values,
                    color_continuous_scale='viridis'
                )
                fig_cat.update_layout(
                    xaxis_title=col,
                    yaxis_title="Count",
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig_cat, use_container_width=True)
    
    # numerical Variables Analysis
    st.markdown("## 🔢 Numerical Variables Analysis")
    
    numerical_cols = ['Agent_Age', 'Agent_Rating']
    if 'real_distance_km' in df.columns:
        numerical_cols.append('real_distance_km')
    
    for col in numerical_cols:
        st.markdown(f"### {col.replace('_', ' ').title()}")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            # Statistics
            stats = df[col].describe()
            st.write(f"**Mean:** {stats['mean']:.2f}")
            st.write(f"**Median:** {stats['50%']:.2f}")
            st.write(f"**Std:** {stats['std']:.2f}")
            st.write(f"**Range:** {stats['min']:.2f} - {stats['max']:.2f}")
        
        with col2:
            # Distribution
            fig_num = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Distribution", "Box Plot")
            )
            
            fig_num.add_trace(
                go.Histogram(x=df[col], nbinsx=30, name="Distribution"),
                row=1, col=1
            )
            
            fig_num.add_trace(
                go.Box(y=df[col], name="Box Plot"),
                row=1, col=2
            )
            
            fig_num.update_layout(
                title=f"{col} Analysis",
                height=400,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                title_font_color='white'
            )
            st.plotly_chart(fig_num, use_container_width=True)
        
        with col3:
            # outlier detection
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
            st.write(f"**Outliers:** {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
    
    # correlation Analysis
    st.markdown("## 🔗 Correlation Analysis")
    
    # select numerical columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    
    fig_corr = px.imshow(
        correlation_matrix,
        title="Correlation Matrix",
        color_continuous_scale='RdBu',
        aspect="auto"
    )
    fig_corr.update_layout(
        height=600,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        title_font_color='white'
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Bivariate Analysis
    st.markdown("## 📈 Bivariate Analysis")
    
    # Delivery time vs categorical variables
    categorical_analysis = st.selectbox(
        "Select categorical variable to analyze with Delivery Time:",
        categorical_cols
    )
    
    fig_box = px.box(
        df, 
        x=categorical_analysis, 
        y='Delivery_Time',
        title=f"Delivery Time by {categorical_analysis}",
        color=categorical_analysis
    )
    fig_box.update_layout(
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        title_font_color='white'
    )
    st.plotly_chart(fig_box, use_container_width=True)
    
    # Geographic Analysis
    if all(col in df.columns for col in ['Store_Latitude', 'Store_Longitude', 'Drop_Latitude', 'Drop_Longitude']):
        st.markdown("## 🗺️ Geographic Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Store locations
            fig_store = px.scatter_mapbox(
                df.sample(1000) if len(df) > 1000 else df,
                lat="Store_Latitude",
                lon="Store_Longitude",
                color="Delivery_Time",
                title="Store Locations (Sample)",
                mapbox_style="open-street-map",
                height=500,
                zoom=10
            )
            st.plotly_chart(fig_store, use_container_width=True)
        
        with col2:
            # Delivery locations
            fig_drop = px.scatter_mapbox(
                df.sample(1000) if len(df) > 1000 else df,
                lat="Drop_Latitude",
                lon="Drop_Longitude",
                color="Delivery_Time",
                title="Delivery Locations (Sample)",
                mapbox_style="open-street-map",
                height=500,
                zoom=10
            )
            st.plotly_chart(fig_drop, use_container_width=True)

# --- Page: Delivery Prediction ---
def render_prediction_page():
    """Render the delivery prediction page with enhanced UI"""
    st.markdown("# 🎯 Smart Delivery Time Prediction")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1.5rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;">
        <h3>🚀 AI-Powered Delivery Predictions</h3>
        <p>Enter order details below to get accurate delivery time predictions with real-world route visualization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input Form
    with st.form("prediction_form"):
        st.markdown("## 📝 Order Information")
        
        # Order Details
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📦 Order Details")
            order_id = st.text_input("Order ID", value=f"ORD_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
            order_date = st.date_input("Order Date", datetime.date.today())
            order_time_str = st.text_input("Order Time (HH:MM:SS)", "19:00:00")
            pickup_time_str = st.text_input("Pickup Time (HH:MM:SS)", "19:15:00")
        
        with col2:
            st.markdown("### 👤 Agent Details")
            agent_age = st.slider("Agent Age", min_value=18, max_value=60, value=28)
            agent_rating = st.slider("Agent Rating", min_value=1.0, max_value=5.0, value=4.5, step=0.1)
        
        # Location Details
        st.markdown("### 📍 Location Details")
        
        # Preset locations for easy testing
        preset_locations = {
            "Bangalore - Koramangala to Whitefield": {
                "store": (12.9351, 77.6245),
                "drop": (12.9698, 77.7500)
            },
            "Bangalore - MG Road to Electronic City": {
                "store": (12.9716, 77.5946),
                "drop": (12.8456, 77.6603)
            },
            "Mumbai - Bandra to Andheri": {
                "store": (19.0596, 72.8295),
                "drop": (19.1136, 72.8697)
            },
            "Delhi - Connaught Place to Gurgaon": {
                "store": (28.6315, 77.2167),
                "drop": (28.4595, 77.0266)
            },
            "Custom Location": {
                "store": (12.9716, 77.5946),
                "drop": (12.9279, 77.6271)
            }
        }
        
        location_preset = st.selectbox("Choose Preset Location or Custom:", list(preset_locations.keys()))
        
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("#### 🏪 Store Location")
            if location_preset != "Custom Location":
                store_lat, store_lon = preset_locations[location_preset]["store"]
            else:
                store_lat, store_lon = 12.9716, 77.5946
            
            store_latitude = st.number_input("Store Latitude", value=float(store_lat), format="%.6f")
            store_longitude = st.number_input("Store Longitude", value=float(store_lon), format="%.6f")
        
        with col4:
            st.markdown("#### 🏠 Delivery Location")
            if location_preset != "Custom Location":
                drop_lat, drop_lon = preset_locations[location_preset]["drop"]
            else:
                drop_lat, drop_lon = 12.9279, 77.6271
                
            drop_latitude = st.number_input("Drop Latitude", value=float(drop_lat), format="%.6f")
            drop_longitude = st.number_input("Drop Longitude", value=float(drop_lon), format="%.6f")
        
        # Contextual Information
        st.markdown("### 🌟 Contextual Information")
        col5, col6, col7 = st.columns(3)
        
        with col5:
            traffic = st.selectbox("🚦 Traffic Condition", ["Low", "Medium", "High", "Jam"])
            weather = st.selectbox("🌤️ Weather", ["Sunny", "Cloudy", "Rainy", "Foggy", "Stormy"])
        
        with col6:
            vehicle = st.selectbox("🏍️ Vehicle Type", ["motorcycle", "scooter", "bicycle"])
            area = st.selectbox("🏙️ Area Type", ["Urban", "Metropolitian", "Semi-Urban"])
        
        with col7:
            category = st.selectbox("📦 Order Category", 
                                   ["Food", "Electronics", "Clothing", "Grocery", "Cosmetics", "Toys", "Sports"])
        
        # Submit button
        submitted = st.form_submit_button(
            "🚀 Predict Delivery Time & Visualize Route", 
            use_container_width=True
        )
    
    # Process prediction
    if submitted:
        # Construct user input
        user_input = {
            'order_id': order_id,
            'order_date': str(order_date),
            'order_time': order_time_str,
            'pickup_time': pickup_time_str,
            'store_latitude': store_latitude,
            'store_longitude': store_longitude,
            'drop_latitude': drop_latitude,
            'drop_longitude': drop_longitude,
            'agent_age': agent_age,
            'agent_rating': agent_rating,
            'traffic': traffic,
            'weather': weather,
            'vehicle': vehicle, # Pass vehicle type for speed limit
            'area': area,
            'category': category
        }
        
        # Show loading
        with st.spinner("🔮 Making prediction and fetching real-time route data..."):
            # Get prediction and map data including vehicle type
            predicted_time, map_data_df = get_prediction(user_input)

        if predicted_time is not None and map_data_df is not None:
            # Success display
            st.markdown(f"""
            <div class="prediction-result">
                <h2>🎉 Prediction Successful!</h2>
                <h1>⏱️ {predicted_time:.1f} minutes</h1>
                <p>Estimated delivery time for Order ID: {order_id}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📏 Route Distance (km)", f"{map_data_df['Real_Distance_km'].iloc[0]:.2f}")
            with col2:
                # Calculate implied average speed from predicted time and real distance
                if predicted_time > 0:
                    implied_speed = (map_data_df['Real_Distance_km'].iloc[0] / (predicted_time / 60))
                    st.metric("⚡ Avg Speed (km/hr)", f"{implied_speed:.2f}")
                else:
                    st.metric("⚡ Avg Speed (km/hr)", "N/A")
            with col3:
                st.metric("🚗 Vehicle Type", map_data_df['Vehicle'].iloc[0])

            st.markdown("---")
            st.markdown("## 🗺️ Real-time Route Visualization")
            
            # visualize route on map
            map_output_path = "outputs/delivery_routes_map.html"
            # Pass the predicted time for the popup
            map_success = visualize_delivery_routes_on_map(
                df=map_data_df,
                pickup_lat_col='Store_Latitude',
                pickup_lon_col='Store_Longitude',
                delivery_lat_col='Drop_Latitude',
                delivery_lon_col='Drop_Longitude',
                popup_cols=[
                    'Order_ID', 'Vehicle', 'Traffic', 'Weather', 'Real_Distance_km'
                ], # Only OSRM distance, not time. Predicted time passed separately.
                map_title=f"Delivery Route for Order ID: {order_id}",
                output_html_path=map_output_path,
                use_real_roads=True,
                show_waypoints=True
            )
            
            if map_success:
                try:
                    with open(map_output_path, 'r') as f:
                        map_html = f.read()
                    
                    # Embed the map in Streamlit with full width and increased height
                    st.components.v1.html(map_html, width=1000, height=800, scrolling=True) 
                    
                    # Download and additional options
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.download_button(
                            label="📥 Download Route Map",
                            data=map_html,
                            file_name=f"route_map_{order_id}.html",
                            mime="text/html",
                            use_container_width=True,
                            key="download_map_btn"
                        )
                    with col2:
                        if st.button("🔄 Predict Another Route", use_container_width=True, key="predict_another_btn"):
                            st.experimental_rerun()
                    with col3:
                        if st.button("📊 Analyze This Route", use_container_width=True, key="analyze_route_btn"):
                            st.info("Route analysis: This prediction considers real-world factors for maximum accuracy!")
                except FileNotFoundError:
                    st.error("❌ Map file not found. Please check the 'outputs' directory.")
                except Exception as e:
                    st.error(f"❌ An error occurred while displaying the map: {e}")
            else:
                st.error("❌ Failed to generate route visualization.")

            # Prediction Insights
            st.markdown("## 📊 Prediction Insights")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🔍 Input Summary")
                insights_data = {
                    "Traffic Impact": f"🚦 {traffic} traffic condition",
                    "Weather Effect": f"🌤️ {weather} weather",
                    "Vehicle Type": f"🏍️ {vehicle.title()}",
                    "Area Category": f"🏙️ {area}",
                    "Order Type": f"📦 {category}",
                    "Agent Experience": f"👤 Age {agent_age}, Rating {agent_rating}/5"
                }
                for key, value in insights_data.items():
                    st.write(f"**{key}:** {value}")
            with col2:
                st.markdown("### ⚡ Quick Actions")
                if st.button("🔄 Predict Another Route", use_container_width=True, key="sidebar_predict_btn_bottom"): # Unique key
                    st.experimental_rerun()
                if st.button("📊 View Historical Data", use_container_width=True, key="sidebar_eda_btn"):
                    st.info("Navigate to EDA Analysis page to explore historical patterns")
                if st.button("📈 Business Insights", use_container_width=True, key="sidebar_insights_btn"):
                    st.info("Navigate to Business Insights for strategic analytics")
        else:
            st.error("❌ Prediction failed. Please check your inputs and try again.")

# --- Page: Business Insights ---
def render_insights_page():
    """Render business insights and analytics page"""
    st.markdown("# 📈 Business Insights & Analytics")
    df = load_data()
    if df is None:
        return

    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;">
        <h3>💼 Strategic Business Analytics</h3>
        <p>Actionable insights for optimizing delivery operations and improving customer experience</p>
    </div>
    """, unsafe_allow_html=True)

    # Key Metrics Dashboard
    st.markdown("## 🎯 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_delivery = df['Delivery_Time'].mean()
        st.metric("⏱️ Avg Delivery Time", f"{avg_delivery:.1f} min")
    with col2:
        if 'real_distance_km' in df.columns:
            avg_distance = df['real_distance_km'].mean()
            st.metric("📏 Avg Distance", f"{avg_distance:.1f} km")
    with col3:
        agent_satisfaction = df['Agent_Rating'].mean()
        st.metric("👤 Agent Rating", f"{agent_satisfaction:.2f}/5")
    with col4:
        total_orders = len(df)
        st.metric("📦 Total Orders", f"{total_orders:,}")

    # performance by Category
    st.markdown("## 📊 Performance by Category")
    category_performance = df.groupby('Category')['Delivery_Time'].mean().sort_values()
    fig_cat_perf = px.bar(
        x=category_performance.index,
        y=category_performance.values,
        title="Average Delivery Time by Category",
        color=category_performance.values,
        color_continuous_scale='plotly3'
    )
    fig_cat_perf.update_layout(height=400)
    st.plotly_chart(fig_cat_perf, use_container_width=True)

    # Agent Performance
    st.markdown("## 👤 Agent Performance Analysis")
    col1, col2 = st.columns(2)
    with col1:
        agent_age_impact = df.groupby('Agent_Age')['Delivery_Time'].mean()
        fig_age = px.line(
            x=agent_age_impact.index,
            y=agent_age_impact.values,
            title="Avg Delivery Time by Agent Age",
            markers=True
        )
        fig_age.update_layout(height=400)
        st.plotly_chart(fig_age, use_container_width=True)
    with col2:
        agent_rating_impact = df.groupby('Agent_Rating')['Delivery_Time'].mean()
        fig_rating = px.bar(
            x=agent_rating_impact.index,
            y=agent_rating_impact.values,
            title="Avg Delivery Time by Agent Rating",
            color=agent_rating_impact.values,
            color_continuous_scale='Plasma' # Corrected colorscale
        )
        fig_rating.update_layout(height=400)
        st.plotly_chart(fig_rating, use_container_width=True)

    # Impact of External Factors
    st.markdown("## 🌧️🚦 Impact of External Factors")
    col1, col2 = st.columns(2)
    with col1:
        traffic_impact = df.groupby('Traffic')['Delivery_Time'].mean().sort_values(ascending=False)
        fig_traffic = px.bar(
            x=traffic_impact.index,
            y=traffic_impact.values,
            title="Delivery Time by Traffic Condition",
            color=traffic_impact.values,
            color_continuous_scale='Reds' # Corrected colorscale
        )
        fig_traffic.update_layout(height=400)
        st.plotly_chart(fig_traffic, use_container_width=True)
    with col2:
        weather_impact = df.groupby('Weather')['Delivery_Time'].mean().sort_values(ascending=False)
        fig_weather = px.bar(
            x=weather_impact.index,
            y=weather_impact.values,
            title="Delivery Time by Weather Condition",
            color=weather_impact.values,
            color_continuous_scale='Blues' # Corrected colorscale
        )
        fig_weather.update_layout(height=400)
        st.plotly_chart(fig_weather, use_container_width=True)

    # Time-based Analysis
    st.markdown("## ⏰ Time-based Insights")
    # Extract hour from order time
    df['Order_Hour'] = pd.to_datetime(df['Order_Time']).dt.hour
    hourly_patterns = df.groupby('Order_Hour').agg({
        'Delivery_Time': 'mean',
        'Order_ID': 'count'
    }).reset_index()
    hourly_patterns.columns = ['Hour', 'Avg_Delivery_Time', 'Order_Count']

    fig_hourly = make_subplots(specs=[[{"secondary_y": True}]])
    fig_hourly.add_trace(
        go.Bar(x=hourly_patterns['Hour'], y=hourly_patterns['Order_Count'], name="Order Count", marker_color='#764ba2'),
        secondary_y=False,
    )
    fig_hourly.add_trace(
        go.Scatter(x=hourly_patterns['Hour'], y=hourly_patterns['Avg_Delivery_Time'], name="Avg Delivery Time", mode='lines+markers', line=dict(color='#667eea', width=3)),
        secondary_y=True,
    )
    fig_hourly.update_layout(
        title_text="Hourly Order Volume and Average Delivery Time",
        xaxis_title="Hour of Day",
        height=500
    )
    fig_hourly.update_yaxes(title_text="Order Count", secondary_y=False)
    fig_hourly.update_yaxes(title_text="Avg Delivery Time (min)", secondary_y=True)
    st.plotly_chart(fig_hourly, use_container_width=True)

    st.markdown("---")
    st.markdown("### 💡 Actionable Recommendations")
    st.info("""
    - **Optimize Agent Allocation:** Identify peak hours and areas with high delivery times to strategically deploy more agents or offer incentives during these periods.
    - **Dynamic Pricing:** Implement dynamic pricing based on predicted delivery times and traffic conditions to manage customer expectations and demand.
    - **Route Optimization:** Leverage real-time traffic data and OSRM integration to dynamically adjust routes, minimizing delays.
    - **Vehicle Type Analysis:** Analyze performance across vehicle types to understand efficiency and operational costs, potentially adjusting fleet composition.
    - **Weather Preparedness:** Develop contingency plans for adverse weather conditions, including surge pricing or service area adjustments.
    """)

# --- Page: About ---
def render_about_page():
    """Render comprehensive about page with app details, accuracy, and links"""
    st.markdown("# ℹ️ About MilesAhead")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1.5rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;">
        <h3>✨ Unlocking Efficiency in Logistics</h3>
        <p>MilesAhead is an advanced Machine Learning-powered platform designed to revolutionize delivery operations. By predicting delivery times with high accuracy and visualizing optimal routes, we help businesses improve efficiency, reduce costs, and enhance customer satisfaction.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(145deg, #2d3436, #636e72); 
                    color: white; padding: 2rem; border-radius: 15px; text-align: center;">
            <h3>🏆 Model Performance</h3>
            <h1 style="color: #00b894; margin: 10px 0;">84.2%</h1>
            <p><strong>R² Score Accuracy</strong></p>
            <hr style="border-color: #667eea;">
            <p><strong>40,109</strong><br>Training Records</p>
            <p><strong>17</strong><br>Feature Variables</p>
            <p><strong>XGBoost</strong><br>Best Model</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(145deg, #2d3436, #636e72); 
                    color: white; padding: 2rem; border-radius: 15px; text-align: center;">
            <h3>💡 Core Features</h3>
            <ul style="list-style-type: none; padding: 0;">
                <li style="margin-bottom: 10px;">✅ Real-time ETA Predictions</li>
                <li style="margin-bottom: 10px;">✅ OSRM-powered Route Visualization</li>
                <li style="margin-bottom: 10px;">✅ Interactive EDA Dashboards</li>
                <li style="margin-bottom: 10px;">✅ Key Business Insights</li>
                <li style="margin-bottom: 10px;">✅ Scalable ML Pipeline</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Technical Details
    st.markdown("## 🔧 Technical Architecture")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(145deg, #2d3436, #636e72); color: white; padding: 1.5rem; border-radius: 15px; margin: 10px 0;">
            <h4>🧠 Machine Learning</h4>
            <ul>
                <li><strong>XGBoost Regressor</strong> - Primary model</li>
                <li><strong>R² Score:</strong> 0.842+</li>
                <li><strong>Features:</strong> 17 variables</li>
                <li><strong>Training Data:</strong> 40,109 records</li>
                <li><strong>Cross-validation</strong> optimized</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(145deg, #2d3436, #636e72); color: white; padding: 1.5rem; border-radius: 15px; margin: 10px 0;">
            <h4>🗺️ Route Optimization</h4>
            <ul>
                <li><strong>OSRM API</strong> integration</li>
                <li><strong>Real road networks</strong></li>
                <li><strong>Turn-by-turn</strong> routing</li>
                <li><strong>Distance & duration</strong> calculation</li>
                <li><strong>Interactive maps</strong> with Folium</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(145deg, #2d3436, #636e72); color: white; padding: 1.5rem; border-radius: 15px; margin: 10px 0;">
            <h4>📊 Data Pipeline</h4>
            <ul>
                <li><strong>Streamlit</strong> web framework</li>
                <li><strong>Plotly</strong> visualizations</li>
                <li><strong>Pandas</strong> data processing</li>
                <li><strong>Scikit-learn</strong> ML pipeline</li>
                <li><strong>Real-time</strong> predictions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Future Improvements
    st.markdown("## 🚀 Future Improvements")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(145deg, #2d3436, #636e72); color: white; padding: 1.5rem; border-radius: 15px; margin: 10px 0;">
            <h4>🔮 AI Improvements</h4>
            <ul>
                <li>Deep Learning models</li>
                <li>Real-time traffic integration</li>
                <li>Weather API integration</li>
                <li>Multi-modal transportation</li>
                <li>Dynamic route optimization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(145deg, #2d3436, #636e72); color: white; padding: 1.5rem; border-radius: 15px; margin: 10px 0;">
            <h4>📱 Platform Features</h4>
            <ul>
                <li>Mobile application</li>
                <li>Real-time tracking</li>
                <li>Customer notifications</li>
                <li>Agent mobile app</li>
                <li>API endpoints</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(145deg, #2d3436, #636e72); color: white; padding: 1.5rem; border-radius: 15px; margin: 10px 0;">
            <h4>🏢 Business Intelligence</h4>
            <ul>
                <li>Advanced analytics dashboard</li>
                <li>Predictive maintenance</li>
                <li>Cost optimization</li>
                <li>Fleet management</li>
                <li>Customer satisfaction AI</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Developer & Links Section
    st.markdown("## 👨‍💻 Developer & Links")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(145deg, #2d3436, #636e72); color: white; padding: 2rem; border-radius: 15px; text-align: center;">
            <h3>👨‍💻 Built by</h3>
            <h2 style="color: #00b894;">Anurag Chaubey</h2>
            <hr style="border-color: #667eea;">
            <p>Passionate about AI, Machine Learning, and building intelligent systems that solve real-world problems.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(145deg, #2d3436, #636e72); color: white; padding: 2rem; border-radius: 15px;">
            <h3>🔗 Project Links</h3>
            <div style="margin: 15px 0;">
                <a href="https://github.com/anuragchaubey1224/MilesAhead-ML-Based-Delivery-Time-Prediction-and-Route-Optimization" target="_blank" style="text-decoration: none; color: white;">
                    <button style="background-color: #667eea; color: white; padding: 10px 20px; border-radius: 5px; border: none; font-size: 1rem; cursor: pointer; width: 100%;">
                        GitHub Repository
                    </button>
                </a>
            </div>
            <div style="margin: 15px 0;">
                <a href="https://linkedin.com/in/anurag-chaubey-ml" target="_blank" style="text-decoration: none; color: white;">
                    <button style="background-color: #667eea; color: white; padding: 10px 20px; border-radius: 5px; border: none; font-size: 1rem; cursor: pointer; width: 100%;">
                        Connect on LinkedIn
                    </button>
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Acknowledgments
    st.markdown("##  Acknowledgments")
    st.markdown("""
    <div style="background: linear-gradient(145deg, #2d3436, #636e72); color: white; padding: 1.5rem; border-radius: 15px; margin: 20px 0;">
        <p>This project leverages the power of several open-source tools and platforms. We extend our gratitude to:</p>
        <ul>
            <li><strong>Project OSRM</strong> - For providing free routing services</li>
            <li><strong>OpenStreetMap</strong> - For open geographic data</li>
            <li><strong>Streamlit Team</strong> - For the amazing web framework</li>
            <li><strong>Plotly</strong> - For interactive visualization tools</li>
            <li><strong>Open Source Community</strong> - For countless libraries and tools</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # License and Usage
    st.markdown("## 📄 License & Usage")
    st.markdown("""
    <div style="background: linear-gradient(145deg, #2d3436, #636e72); color: white; padding: 1.5rem; border-radius: 15px; margin: 20px 0;">
        <p><strong>This project is open source and available under the MIT License.</strong></p>
        <p>Feel free to use, modify, and distribute this code for educational and commercial purposes. Attribution to the original author is appreciated.</p>
        <h4>🚀 How to Use This Project:</h4>
        <ol>
            <li>Clone the repository from GitHub</li>
            <li>Install required dependencies</li>
            <li>Prepare your delivery dataset</li>
            <li>Train the model with your data</li>
            <li>Deploy the Streamlit application</li>
        </ol>
        <p><em>For detailed setup instructions, please refer to the README.md file in the GitHub repository.</em></p>
    </div>
    """, unsafe_allow_html=True)


# --- Main Application Logic ---
if __name__ == "__main__":
    selected_page = render_sidebar()

    if selected_page == "🏠 Home":
        render_home_page()
    elif selected_page == "📊 EDA Analysis":
        render_eda_page()
    elif selected_page == "🎯 Delivery Prediction":
        render_prediction_page()
    elif selected_page == "📈 Business Insights":
        render_insights_page()
    elif selected_page == "ℹ️ About":
        render_about_page()