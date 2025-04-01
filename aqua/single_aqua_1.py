from ultralytics import YOLO
import cv2
import os
import logging
import numpy as np
from estimator_func import positional_estimate
import time
import matplotlib.pyplot as plt

np.set_printoptions(precision=16, suppress=True)

# Load data
data = np.genfromtxt('single_1.csv', skip_header=1, delimiter=',', dtype=np.float64)

# Suppress ultralytics logs
logging.getLogger("ultralytics").setLevel(logging.WARNING)


# Load YOLO model
model = YOLO("yolo11n_trained.pt").to("cuda")

video_path = "./videos/single_1.mp4"

true_gps_lat = 13.1938678234695	
true_gps_lon = -59.6412682002346

# Compute positional estimate
image_width = 1920
image_height = 1080


repeats= 1
	
# Create output directories
output_dir = "output_videos"
image_output_dir = "output_images"  # Directory for saved images
os.makedirs(output_dir, exist_ok=True)
os.makedirs(image_output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#out = cv2.VideoWriter(f"{output_dir}/output_video1.mp4", fourcc, fps, (width, height))


error = []

# Control variable

tmp_counter = 0

true_gps_lat_list = []
true_gps_lon_list = []
est_lat_list = []
est_lon_list = []

average_time = []

for t in range(repeats):

    accumulated_time= 0
    cap = cv2.VideoCapture(video_path)

    tele_count= 0
    frame_count= 0
    while cap.isOpened():

        start= time.time()

        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)


        for result in results:
            for i, box in enumerate(result.boxes):
                # Copy the original frame for each bounding box
                bound_image = frame.copy()

                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw only this bounding box on bound_image
                #cv2.rectangle(bound_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

                x_center = x1 + abs(x2 - x1)
                y_center = image_height - (y1 + abs(y2 - y1))

                value_error, est_long, est_lat = positional_estimate(
                    data[frame_count][2], data[frame_count][3], data[frame_count][5], data[frame_count][4], 
                    data[frame_count][0], data[frame_count][1], x_center, y_center, 
                    true_gps_lat, true_gps_lon
                )
                """error.append(value_error*1000)
                true_gps_lat_list.append(true_gps_lat)
                true_gps_lon_list.append(true_gps_lon)
                est_lat_list.append(est_lat)
                est_lon_list.append(est_long)"""


        end= time.time()
        accumulated_time += end-start
        # Write the original frame to the output video
        #out.write(frame)
        tele_count += 1
        frame_count= int(tele_count/3)
    
    average_time.append(accumulated_time/frame_count)
    
average_time= np.array(average_time)

print("Average time and std per frame:", np.mean(average_time), np.std(average_time)) 


print("Frames processed:", frame_count)
#print("Min error:", np.min(np.array(error)))
#print("Max error:", np.max(np.array(error)))

cap.release()
#out.release()
#cv2.destroyAllWindows()

"""indices = list(range(len(error[4:34])))


# Write the error and true gps and estimated gps to a file
with open("error_single_1.txt", "w") as f:
    for i in range(5, 35):
        f.write(f"{error[i]}, {true_gps_lat_list[i]}, {true_gps_lon_list[i]}, {est_lat_list[i]}, {est_lon_list[i]}\n")


# Plot bar graph
plt.bar(indices, error[5:35], color='blue', alpha=0.7)

# Labels and title
plt.xlabel("Time Step")
plt.ylabel("Error Distance (meters)")
plt.title("Aqua Localization Sample 1")
plt.ylim(0, 10)
plt.savefig("single_1.png", dpi=300, bbox_inches="tight")"""