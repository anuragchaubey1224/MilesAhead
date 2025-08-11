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
                'duration_minutes': route['duration'] / 60,
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
                         route_info: Dict[str, Any] = None) -> str:
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
        popup_html += f"<b>🛣️ Route Distance:</b> {route_info['distance_km']:.2f} km<br>"
        popup_html += f"<b>⏱️ Estimated Duration:</b> {route_info['duration_minutes']:.1f} min<br>"
    
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

            # skip rows with missing coordinates
            if pd.isna(p_lat) or pd.isna(p_lon) or pd.isna(d_lat) or pd.isna(d_lon):
                continue

            route_count += 1
            route_info = None

            # get real route from OSRM if enabled
            if use_real_roads:
                print(f"Fetching route {route_count} from OSRM...")
                route_info = get_real_route_from_osrm(p_lon, p_lat, d_lon, d_lat)
                time.sleep(0.1)  # Small delay to avoid overwhelming the API

            # create popup text with route information
            pickup_popup = create_enhanced_popup(row, popup_cols, route_info)
            pickup_popup += f"<b>📍 Pickup:</b> ({p_lat:.4f}, {p_lon:.4f})"
            
            delivery_popup = create_enhanced_popup(row, popup_cols, route_info)
            delivery_popup += f"<b>🎯 Delivery:</b> ({d_lat:.4f}, {d_lon:.4f})"

            # add pickup marker (green)
            folium.Marker(
                location=[p_lat, p_lon],
                popup=folium.Popup(pickup_popup, max_width=350),
                icon=folium.Icon(color='green', icon='play', prefix='fa'),
                tooltip=f"🚚 Pickup - Route {route_count}"
            ).add_to(m)

            # Add delivery marker (red)
            folium.Marker(
                location=[d_lat, d_lon],
                popup=folium.Popup(delivery_popup, max_width=350),
                icon=folium.Icon(color='red', icon='stop', prefix='fa'),
                tooltip=f"🏠 Delivery - Route {route_count}"
            ).add_to(m)

            # add route line
            if route_info and route_info.get('geometry'):
                # use real road geometry from OSRM
                route_coords = decode_polyline_to_coordinates(route_info['geometry'])
                
                folium.PolyLine(
                    locations=route_coords,
                    color='#FF6B6B',
                    weight=4,
                    opacity=0.8,
                    popup=folium.Popup(f"<b>Real Route {route_count}</b><br>" + 
                                     create_enhanced_popup(row, popup_cols, route_info), 
                                     max_width=350),
                    tooltip=f"Route {route_count} - {route_info['distance_km']:.2f} km"
                ).add_to(m)
                
                # add waypoints if requested
                if show_waypoints and len(route_coords) > 2:
                    # add small markers for significant waypoints 
                    step = max(1, len(route_coords) // 10)
                    for i in range(step, len(route_coords) - step, step):
                        folium.CircleMarker(
                            location=route_coords[i],
                            radius=3,
                            popup=f"Waypoint {i // step}",
                            color='orange',
                            fill=True,
                            fillColor='orange'
                        ).add_to(m)
                
                successful_routes += 1
            else:
                # fallback to straight line if OSRM fails
                folium.PolyLine(
                    locations=[[p_lat, p_lon], [d_lat, d_lon]],
                    color='blue',
                    weight=2.5,
                    opacity=0.6,
                    dash_array='5, 5',
                    popup=folium.Popup(f"<b>Direct Route {route_count}</b><br>" + 
                                     create_enhanced_popup(row, popup_cols), 
                                     max_width=350),
                    tooltip=f"Direct Route {route_count}"
                ).add_to(m)

        # add enhanced title and legend with full-width styling
        title_html = f'''
        <style>
        .leaflet-container {{
            width: 100% !important;
            height: 100% !important;
        }}
        .folium-map {{
            width: 100% !important;
            height: 100% !important;
        }}
        </style>
        <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%); 
                    width: 90%; z-index: 9999; background: rgba(255,255,255,0.9); 
                    padding: 10px; border-radius: 10px; text-align: center; 
                    box-shadow: 0 2px 6px rgba(0,0,0,0.3);">
            <h3 style="margin: 0; color: #2E4057; font-size: 18px;">
                <b>{map_title}</b>
            </h3>
            <p style="margin: 5px 0 0 0; font-size: 12px; color: #666;">
                📍 {route_count} routes | ✅ {successful_routes} real road paths | 
                🚚 Green: Pickup | 🏠 Red: Delivery
            </p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))

        # add legend
        legend_html = '''
        <div style="position: fixed; bottom: 50px; right: 50px; width: 200px; height: 120px; 
                    background: rgba(255,255,255,0.9); border: 2px solid grey; z-index: 9999; 
                    font-size: 12px; padding: 10px; border-radius: 5px;">
            <p style="margin: 0; font-weight: bold;">🗺️ Map Legend</p>
            <p style="margin: 5px 0;"><span style="color: green;">●</span> Pickup Location</p>
            <p style="margin: 5px 0;"><span style="color: red;">●</span> Delivery Location</p>
            <p style="margin: 5px 0;"><span style="color: #FF6B6B;">━</span> Real Road Route</p>
            <p style="margin: 5px 0;"><span style="color: blue;">╌</span> Direct Route</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))

        # save map
        m.save(output_html_path)
        print(f"✅ Route map saved successfully: {output_html_path}")
        print(f"📊 Summary: {successful_routes}/{route_count} routes used real road data")
        return True
        
    except Exception as e:
        print(f"Error creating route map: {e}")
        import traceback
        traceback.print_exc()
        return False

def visualize_multiple_deliveries_map(
    df: pd.DataFrame,
    depot_lat: float,
    depot_lon: float,
    delivery_lat_col: str,
    delivery_lon_col: str,
    popup_cols: List[str] = None,
    zoom_start: int = 12,
    map_title: str = "Multiple Delivery Optimization",
    output_html_path: str = "outputs/multiple_deliveries_map.html"
) -> bool:
    """
    Visualizes multiple delivery routes from a single depot/store location.
    Useful for route optimization visualization.
    """
    try:
        if df.empty:
            print("Error: DataFrame is empty")
            return False

        # create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_html_path), exist_ok=True)

        # create map centered on depot
        m = folium.Map(
            location=[depot_lat, depot_lon], 
            zoom_start=zoom_start,
            control_scale=True,
            tiles='OpenStreetMap'
        )

        # add depot marker
        folium.Marker(
            location=[depot_lat, depot_lon],
            popup=f"<b>🏢 Depot/Store</b><br>Coordinates: ({depot_lat:.4f}, {depot_lon:.4f})",
            icon=folium.Icon(color='black', icon='home', prefix='fa'),
            tooltip="Main Depot"
        ).add_to(m)

        # add delivery routes
        for idx, row in df.iterrows():
            d_lat, d_lon = row[delivery_lat_col], row[delivery_lon_col]
            
            if pd.isna(d_lat) or pd.isna(d_lon):
                continue

            # get route information
            route_info = get_real_route_from_osrm(depot_lon, depot_lat, d_lon, d_lat)
            
            # create popup
            popup_text = create_enhanced_popup(row, popup_cols, route_info)
            popup_text += f"<b>🎯 Delivery {idx + 1}:</b> ({d_lat:.4f}, {d_lon:.4f})"

            # add delivery marker
            folium.Marker(
                location=[d_lat, d_lon],
                popup=folium.Popup(popup_text, max_width=350),
                icon=folium.Icon(color='red', icon='gift', prefix='fa'),
                tooltip=f"Delivery {idx + 1}"
            ).add_to(m)

            # add route
            if route_info and route_info.get('geometry'):
                route_coords = decode_polyline_to_coordinates(route_info['geometry'])
                folium.PolyLine(
                    locations=route_coords,
                    color=f'#{hash(str(idx)) % 16777216:06x}',  # Random color per route
                    weight=3,
                    opacity=0.7,
                    popup=folium.Popup(f"Route to Delivery {idx + 1}", max_width=200)
                ).add_to(m)

        # add title
        title_html = f'''
        <h3 align="center" style="font-size:20px; margin-top:10px; color: #2E4057;">
            <b>{map_title}</b>
        </h3>
        '''
        m.get_root().html.add_child(folium.Element(title_html))

        m.save(output_html_path)
        print(f"✅ Multiple delivery map saved: {output_html_path}")
        return True
        
    except Exception as e:
        print(f"Error creating multiple delivery map: {e}")
        return False

