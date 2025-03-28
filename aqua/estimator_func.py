import math
import requests
import numpy as np
import numpy as np
from numba import jit

# Precompute sensor parameters globally if they don't change:
sensor_width = 36.0
# Image and camera parameters
#sensor_height = sensor_width * (image_height / image_width)  # Maintain aspect ratio

# Earth's approximate conversions
feet_per_degree_lat = 364000  # Approximate feet per degree of latitude
#fov_x = 2 * np.degrees(np.arctan((sensor_width / 2) / focal_length))
#fov_y = 2 * np.degrees(np.arctan((sensor_height / 2) / focal_length))


# Compute positional estimate
image_width = 1920
image_height = 1080

focal_length=  20.3 # Focal length in mm




@jit(nopython=True, fastmath=True)
def haversine(lat1, lon1, lat2, lon2):
    # Convert degrees to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    # Compute deltas
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Earth radius (mean radius in kilometers)
    R = 6371.0

    # Calculate and return distance
    return R * c

@jit(nopython=True, fastmath=True)
def positional_estimate_multi(altitude,drone_angle,camera_angle,heading,lat,lon,pixel_x,pixel_y, true_lat,true_lon):
    camera_angle = 90 + camera_angle

    camera_angle += drone_angle
    sensor_height = sensor_width * (image_height / image_width)  # Maintain aspect ratio

    
    feet_per_degree_lon = feet_per_degree_lat * np.cos(np.radians(lat))  # Adjusted for latitude

    # Compute FoV (Field of View) for horizontal and vertical axes
    fov_x = 2 * np.degrees(np.arctan((sensor_width / 2) / focal_length))
    fov_y = 2 * np.degrees(np.arctan((sensor_height / 2) / focal_length))

    # Compute angular displacement from center
    delta_x = pixel_x - (image_width / 2)  # Left/right shift
    delta_y = pixel_y - (image_height / 2)  # Up/down shift

    theta_x = np.arctan((2*delta_x/altitude)*np.tan(fov_x/2))
    theta_y = np.arctan((2*delta_y/altitude)*np.tan(fov_y/2))

    # Compute new ground distance
    ground_distance_object = altitude * np.tan(np.radians(camera_angle + theta_y))

    # Compute lateral displacement due to theta_x
    ground_distance_x = np.sqrt(ground_distance_object**2 + altitude**2) * np.tan(np.radians(theta_x))

    # Compute new offsets including lateral shift
    delta_north_object = ground_distance_object * np.cos(np.radians(heading))
    delta_east_object = ground_distance_object * np.sin(np.radians(heading))

    # Adjust for lateral shift due to theta_x
    delta_east_object += ground_distance_x * np.cos(np.radians(heading + 90))
    delta_north_object += ground_distance_x * np.sin(np.radians(heading + 90))

    # Convert feet to degrees
    delta_lat_object = delta_north_object / feet_per_degree_lat
    delta_lon_object = delta_east_object / feet_per_degree_lon

    # Compute final GPS coordinates
    object_lat_final = lat + delta_lat_object
    object_lon_final = lon + delta_lon_object

    #print("Object GPS Coordinates:", object_lat_final, object_lon_final)
    errors= np.array([haversine(object_lat_final, object_lon_final, true_lat[i], true_lon[i]) for i in range(len(true_lat))])
    return np.min(errors), object_lon_final, object_lat_final



@jit(nopython=True, fastmath=True)
def positional_estimate(altitude,drone_angle,camera_angle,heading,lat,lon,pixel_x,pixel_y, true_lat,true_lon):
    
    camera_angle = 90 + camera_angle+ drone_angle

    feet_per_degree_lon = feet_per_degree_lat * np.cos(np.radians(lat))  # Adjusted for latitude

    sensor_height = sensor_width * (image_height / image_width)  # Maintain aspect ratio

    # Compute FoV (Field of View) for horizontal and vertical axes
    fov_x = 2 * np.degrees(np.arctan((sensor_width / 2) / focal_length))
    fov_y = 2 * np.degrees(np.arctan((sensor_height / 2) / focal_length))

    # Compute angular displacement from center
    delta_x = pixel_x - (image_width / 2)  # Left/right shift
    delta_y = pixel_y - (image_height / 2)  # Up/down shift

    theta_x = np.arctan((2*delta_x/altitude)*np.tan(fov_x/2))# Horizontal angle shift
    theta_y = np.arctan((2*delta_y/altitude)*np.tan(fov_y/2))# Vertical angle shift

    # Compute new ground distance
    ground_distance_object = altitude * np.tan(np.radians(camera_angle + theta_y))

    # Compute lateral displacement due to theta_x
    ground_distance_x = np.sqrt(ground_distance_object**2 + altitude**2) * np.tan(np.radians(theta_x))
    
    # Compute new offsets including lateral shift
    delta_north_object = ground_distance_object * np.cos(np.radians(heading))
    delta_east_object = ground_distance_object * np.sin(np.radians(heading))

    # Adjust for lateral shift due to theta_x
    delta_east_object += ground_distance_x * np.cos(np.radians(heading + 90))
    delta_north_object += ground_distance_x * np.sin(np.radians(heading + 90))

    # Convert ground displacement from feet to degrees.
    delta_lat_object = delta_north_object / feet_per_degree_lat
    delta_lon_object = delta_east_object / feet_per_degree_lon

    # Calculate final GPS coordinates.
    object_lat_final = lat + delta_lat_object
    object_lon_final = lon + delta_lon_object

    error = haversine(object_lat_final, object_lon_final, true_lat, true_lon)
    return error, object_lon_final, object_lat_final

def pass_image_to_model(image_path, model_url):
    # Load the image from the file
    with open(image_path, 'rb') as img_file:
        image_data = img_file.read()
    
    # You might also need headers or other metadata depending on the model
    headers = {'Content-Type': 'application/octet-stream'}
    
    # Send a POST request to the model API (assuming a REST API is used)
    response = requests.post(model_url, headers=headers, files={'image': image_data})

    # Check if the request was successful
    if response.status_code == 200:
        print("Success! Model output:", response.json())
    else:
        print("Error:", response.status_code, response.text)

