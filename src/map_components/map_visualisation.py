# map_visualisation.py

import folium
import pandas as pd
import os
import requests
import json
import time
from typing import Optional, List, Tuple, Dict, Any

# OSRM API configuration
OSRM_ROUTE_URL = "http://router.project-osrm.org/route/v1/driving/{},{};{},{}?overview=full&geometries=geojson"
OSRM_TABLE_URL = "http://router.project-osrm.org/table/v1/driving/{}"

def get_real_route_from_osrm(pickup_lon: float, pickup_lat: float, 
                           drop_lon: float, drop_lat: float) -> Optional[Dict[str, Any]]:
    """
    fetches the real driving route from the OSRM API including geometry.
    returns route information including distance, duration, and geometry.
    """
    try:
        url = OSRM_ROUTE_URL.format(pickup_lon, pickup_lat, drop_lon, drop_lat)
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data['code'] == 'Ok' and 'routes' in data and data['routes']:
            route = data['routes'][0]
            return {
                'distance_km': route['distance'] / 1000,
                'duration_minutes': route['duration'] / 60, # Keep for internal use if needed, but won't show in popup now
                'geometry': route['geometry']['coordinates'],
                'waypoints': data.get('waypoints', [])
            }
        else:
            print(f"Error from OSRM API: {data.get('code', 'Unknown error')}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Failed to connect to OSRM server: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during OSRM call: {e}")
        return None

def decode_polyline_to_coordinates(geometry: List[List[float]]) -> List[List[float]]:
    """
    convert OSRM geometry coordinates to lat/lon format for Folium.
    OSRM returns coordinates as [lon, lat], but Folium expects [lat, lon].
    """
    return [[coord[1], coord[0]] for coord in geometry]

def create_enhanced_popup(row: pd.Series, popup_cols: List[str] = None, 
                         route_info: Dict[str, Any] = None, predicted_time: Optional[float] = None) -> str:
    """
    create an enhanced popup with route information and custom columns
    """
    popup_html = "<div style='font-family: Arial, sans-serif; max-width: 300px;'>"
    
    # add custom columns first
    if popup_cols:
        for col in popup_cols:
            if col in row and not pd.isna(row[col]):
                value = row[col]
                if isinstance(value, float):
                    if 'time' in col.lower() or 'duration' in col.lower():
                        # Exclude other 'time' columns if they are not the predicted time
                        if col != 'Predicted_Delivery_Time':
                            continue 
                        popup_html += f"<b>{col.title()}:</b> {value:.2f} min<br>"
                    elif 'distance' in col.lower():
                        popup_html += f"<b>{col.title()}:</b> {value:.2f} km<br>"
                    elif 'rating' in col.lower():
                        popup_html += f"<b>{col.title()}:</b> {value:.1f}/5<br>"
                    else:
                        popup_html += f"<b>{col.title()}:</b> {value:.2f}<br>"
                else:
                    popup_html += f"<b>{col.title()}:</b> {value}<br>"
    
    # add route information if available
    if route_info:
        popup_html += "<hr style='margin: 5px 0;'>"
        popup_html += f"<b>🛣️ Route Distance (OSRM):</b> {route_info['distance_km']:.2f} km<br>"
    
    # Add predicted time from the model explicitly
    if predicted_time is not None:
        if route_info: # Add a separator if route info was present
            popup_html += "<hr style='margin: 5px 0;'>" 
        popup_html += f"<b>⏱️ Predicted Time (Model):</b> {predicted_time:.1f} min<br>"

    popup_html += "</div>"
    return popup_html

def visualize_locations_on_map(
    df: pd.DataFrame,
    latitude_col: str,
    longitude_col: str,
    popup_cols: List[str] = None,
    zoom_start: int = 12,
    map_title: str = "Location Visualization",
    output_html_path: str = "outputs/map_output.html"
) -> bool:
    """
    visualizes individual locations on a Folium map with enhanced features.
    
    args:
        df: DataFrame containing location data
        latitude_col: Column name for latitude
        longitude_col: Column name for longitude
        popup_cols: List of columns to include in popup
        zoom_start: Initial zoom level
        map_title: Title for the map
        output_html_path: Path to save the HTML file
        
    returns:
        bool: True if successful, False otherwise
    """
    try:
        if df.empty:
            print("Error: DataFrame is empty")
            return False

        if latitude_col not in df.columns or longitude_col not in df.columns:
            print(f"Error: Required columns '{latitude_col}' or '{longitude_col}' missing")
            return False

        # create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_html_path), exist_ok=True)

        # define map center based on the first non-null location
        valid_locations = df.dropna(subset=[latitude_col, longitude_col])
        if valid_locations.empty:
            print("Error: No valid coordinates found")
            return False
            
        center_lat = valid_locations[latitude_col].iloc[0]
        center_lon = valid_locations[longitude_col].iloc[0]
        
        # create map with enhanced styling
        m = folium.Map(
            location=[center_lat, center_lon], 
            zoom_start=zoom_start,
            control_scale=True,
            tiles='OpenStreetMap'
        )

        # add markers for each location
        for idx, row in valid_locations.iterrows():
            lat, lon = row[latitude_col], row[longitude_col]
            
            popup_text = create_enhanced_popup(row, popup_cols)
            popup_text += f"<b>📍 Coordinates:</b> ({lat:.4f}, {lon:.4f})"

            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_text, max_width=350),
                tooltip=f"Location {idx + 1}",
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(m)

        # add title to the map
        title_html = f'''
        <h3 align="center" style="font-size:20px; margin-top:10px; color: #2E4057;">
            <b>{map_title}</b>
        </h3>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # save map
        m.save(output_html_path)
        print(f"✅ Map saved successfully: {output_html_path}")
        return True
        
    except Exception as e:
        print(f"Error creating location map: {e}")
        return False

def visualize_delivery_routes_on_map(
    df: pd.DataFrame,
    pickup_lat_col: str,
    pickup_lon_col: str,
    delivery_lat_col: str,
    delivery_lon_col: str,
    popup_cols: List[str] = None,
    zoom_start: int = 12,
    map_title: str = "Delivery Route Visualization",
    output_html_path: str = "outputs/delivery_routes_map.html",
    use_real_roads: bool = True,
    show_waypoints: bool = True
) -> bool:
    """
    visualizes delivery routes with real road paths using OSRM API.
    
    args:
        df: DataFrame containing route data
        pickup_lat_col: Column name for pickup latitude
        pickup_lon_col: Column name for pickup longitude
        delivery_lat_col: Column name for delivery latitude
        delivery_lon_col: Column name for delivery longitude
        popup_cols: List of columns to include in popup
        zoom_start: Initial zoom level
        map_title: Title for the map
        output_html_path: Path to save the HTML file
        use_real_roads: Whether to use OSRM for real road routing
        show_waypoints: Whether to show intermediate waypoints
        
    returns:
        bool: True if successful, False otherwise
    """
    try:
        if df.empty:
            print("Error: DataFrame is empty")
            return False

        required_cols = [pickup_lat_col, pickup_lon_col, delivery_lat_col, delivery_lon_col]
        if not all(col in df.columns for col in required_cols):
            print(f"Error: Missing required columns: {required_cols}")
            return False

        # create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_html_path), exist_ok=True)

        # calculate map center based on all coordinates
        all_lats = pd.concat([df[pickup_lat_col], df[delivery_lat_col]]).dropna()
        all_lons = pd.concat([df[pickup_lon_col], df[delivery_lon_col]]).dropna()
        if all_lats.empty or all_lons.empty:
            print("Error: No valid coordinates found")
            return False
        center_lat = all_lats.mean()
        center_lon = all_lons.mean()

        # create map with enhanced styling
        m = folium.Map(
            location=[center_lat, center_lon], 
            zoom_start=zoom_start,
            control_scale=True,
            tiles='OpenStreetMap'
        )

        # add routes for each row
        route_count = 0
        successful_routes = 0
        for idx, row in df.iterrows():
            p_lat, p_lon = row[pickup_lat_col], row[pickup_lon_col]
            d_lat, d_lon = row[delivery_lat_col], row[delivery_lon_col]
            predicted_time = row.get('Predicted_Delivery_Time') # Get predicted time for popup

            # skip rows with missing coordinates
            if pd.isna(p_lat) or pd.isna(p_lon) or pd.isna(d_lat) or pd.isna(d_lon):
                continue

            route_count += 1
            route_info = None

            # get real route from OSRM if enabled
            if use_real_roads:
                print(f"Fetching route {route_count} from OSRM...")
                route_info = get_real_route_from_osrm(p_lon, p_lat, d_lon, d_lat)
                time.sleep(0.1) # Small delay to avoid overwhelming the API

            # create popup text with route information
            # Pass the predicted_time to the popup creation function
            pickup_popup = create_enhanced_popup(row, popup_cols, route_info, predicted_time)

            # add pickup and drop markers
            folium.Marker(
                location=[p_lat, p_lon],
                popup=folium.Popup(pickup_popup, max_width=350),
                tooltip=f"Pickup for Order {row.get('Order_ID', idx+1)}",
                icon=folium.Icon(color='green', icon='play')
            ).add_to(m)

            folium.Marker(
                location=[d_lat, d_lon],
                popup=folium.Popup(pickup_popup, max_width=350),
                tooltip=f"Drop-off for Order {row.get('Order_ID', idx+1)}",
                icon=folium.Icon(color='red', icon='stop')
            ).add_to(m)

            if route_info and route_info['geometry']:
                # decode and add route polyline
                route_coords = decode_polyline_to_coordinates(route_info['geometry'])
                folium.PolyLine(
                    locations=route_coords,
                    color='blue',
                    weight=5,
                    opacity=0.7,
                    tooltip=f"Route: {route_info['distance_km']:.2f} km"
                ).add_to(m)
                successful_routes += 1
            elif not use_real_roads:
                # simple straight line if not using real roads
                folium.PolyLine(
                    locations=[[p_lat, p_lon], [d_lat, d_lon]],
                    color='gray',
                    weight=3,
                    opacity=0.5,
                    dash_array='5, 5',
                    tooltip="Direct Line (OSRM disabled)"
                ).add_to(m)
                successful_routes += 1
            else:
                print(f"Could not get route for Order ID: {row.get('Order_ID', 'N/A')}. Displaying direct line.")
                # Fallback to straight line if OSRM fails
                folium.PolyLine(
                    locations=[[p_lat, p_lon], [d_lat, d_lon]],
                    color='orange',
                    weight=3,
                    opacity=0.5,
                    dash_array='5, 5',
                    tooltip="Direct Line (OSRM failed)"
                ).add_to(m)
                successful_routes += 1


        # add title to the map
        title_html = f'''
        <h3 align="center" style="font-size:20px; margin-top:10px; color: #2E4057;">
            <b>{map_title}</b>
        </h3>
        '''
        m.get_root().html.add_child(folium.Element(title_html))

        if successful_routes > 0:
            # save map
            m.save(output_html_path)
            print(f"✅ Map with {successful_routes} routes saved successfully: {output_html_path}")
            return True
        else:
            print("No successful routes to visualize.")
            return False
        
    except Exception as e:
        print(f"Error creating delivery routes map: {e}")
        return False