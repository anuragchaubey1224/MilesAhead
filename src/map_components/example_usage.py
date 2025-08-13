#!/usr/bin/env python3
"""
Example usage of the enhanced map visualization functions.
This script demonstrates how to use the new map visualization features
with real road routing using OSRM API.
"""

import pandas as pd
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from map_components.map_visualisation import (
    visualize_locations_on_map,
    visualize_delivery_routes_on_map,
    visualize_multiple_deliveries_map
)

def create_sample_data():
    """Create sample delivery data for demonstration."""
    
    # Sample delivery data with coordinates in Bangalore, India
    sample_data = {
        'order_id': ['ORD_001', 'ORD_002', 'ORD_003'],
        'store_latitude': [12.9716, 12.9716, 12.9716],  # Common store location
        'store_longitude': [77.5946, 77.5946, 77.5946],
        'drop_latitude': [12.9279, 12.9351, 12.9698],   # Different delivery locations
        'drop_longitude': [77.6271, 77.6245, 77.7500],
        'predicted_time': [25.5, 18.2, 35.8],
        'real_distance_km': [8.2, 6.1, 12.5],
        'agent_age': [28, 32, 25],
        'agent_rating': [4.7, 4.2, 4.9],
        'traffic': ['High', 'Medium', 'Low'],
        'vehicle': ['Motorcycle', 'Car', 'Motorcycle']
    }
    
    return pd.DataFrame(sample_data)

def demo_location_visualization():
    """Demonstrate basic location visualization."""
    print("🗺️  Creating location visualization demo...")
    
    # Sample location data
    locations_data = {
        'location_name': ['Store A', 'Store B', 'Store C'],
        'latitude': [12.9716, 12.9351, 12.9698],
        'longitude': [77.5946, 77.6245, 77.7500],
        'store_type': ['Main', 'Branch', 'Outlet'],
        'rating': [4.5, 4.2, 4.8]
    }
    
    df_locations = pd.DataFrame(locations_data)
    
    success = visualize_locations_on_map(
        df=df_locations,
        latitude_col='latitude',
        longitude_col='longitude',
        popup_cols=['location_name', 'store_type', 'rating'],
        map_title="Store Locations Demo",
        output_html_path="outputs/demo_locations.html"
    )
    
    if success:
        print("✅ Location visualization created successfully!")
    else:
        print("❌ Failed to create location visualization")

def demo_delivery_routes():
    """Demonstrate delivery route visualization with real road paths."""
    print("🚚 Creating delivery routes visualization demo...")
    
    df_routes = create_sample_data()
    
    success = visualize_delivery_routes_on_map(
        df=df_routes,
        pickup_lat_col='store_latitude',
        pickup_lon_col='store_longitude',
        delivery_lat_col='drop_latitude',
        delivery_lon_col='drop_longitude',
        popup_cols=['order_id', 'predicted_time', 'real_distance_km', 'traffic', 'vehicle'],
        map_title="Real Road Delivery Routes Demo",
        output_html_path="outputs/demo_delivery_routes.html",
        use_real_roads=True,
        show_waypoints=True
    )
    
    if success:
        print("✅ Delivery routes visualization created successfully!")
    else:
        print("❌ Failed to create delivery routes visualization")

def demo_multiple_deliveries():
    """Demonstrate multiple deliveries from single depot."""
    print("📦 Creating multiple deliveries visualization demo...")
    
    df_routes = create_sample_data()
    
    # Use the first store location as depot
    depot_lat = df_routes['store_latitude'].iloc[0]
    depot_lon = df_routes['store_longitude'].iloc[0]
    
    success = visualize_multiple_deliveries_map(
        df=df_routes,
        depot_lat=depot_lat,
        depot_lon=depot_lon,
        delivery_lat_col='drop_latitude',
        delivery_lon_col='drop_longitude',
        popup_cols=['order_id', 'predicted_time', 'agent_rating'],
        map_title="Multiple Deliveries from Single Depot",
        output_html_path="outputs/demo_multiple_deliveries.html"
    )
    
    if success:
        print("✅ Multiple deliveries visualization created successfully!")
    else:
        print("❌ Failed to create multiple deliveries visualization")

def main():
    """Run all demonstrations."""
    print("🎯 Starting Map Visualization Demonstrations\n")
    
    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)
    
    # Run demonstrations
    demo_location_visualization()
    print()
    
    demo_delivery_routes()
    print()
    
    demo_multiple_deliveries()
    print()
    
    print("🎉 All demonstrations completed!")
    print("\n📁 Check the following files in the 'outputs' directory:")
    print("   - demo_locations.html")
    print("   - demo_delivery_routes.html") 
    print("   - demo_multiple_deliveries.html")
    print("\n💡 Open these HTML files in your browser to view the interactive maps!")

if __name__ == "__main__":
    main()
