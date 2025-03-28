from ultralytics import YOLO
import cv2
import os
import logging
import numpy as np
from estimator_func import positional_estimate_multi
import time
import matplotlib.pyplot as plt

np.set_printoptions(precision=16, suppress=True)

# Load data
data = np.genfromtxt('multi_2.csv', skip_header=1, delimiter=',', dtype=np.float64)

# Suppress ultralytics logs
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# Load YOLO model
model = YOLO("yolo11n_multiaqua.pt")

video_path = "./videos/multi_2.mp4"

true_gps_lat = [13.1937833853264,13.193781692443,13.1937807194586]
true_gps_lon = [-59.6418223391455,-59.6418205112973,-59.6418110324059]

# Compute positional estimate
image_width = 1920
image_height = 1080

repeats= 50
	
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
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(f"{output_dir}/output_video1.mp4", fourcc, fps, (width, height))

frame_count = 0
error = [[],[],[]]
true_gps_lat_list = [[],[],[]]
true_gps_lon_list = [[],[],[]]

est_lat_list = [[],[],[]]
est_lon_list = [[],[],[]]



average_time = []

for t in range(repeats):
    accumulated_time= 0
    cap = cv2.VideoCapture(video_path)

    tele_count= 0
    frame_count= 0
    while cap.isOpened():

        start= time.time()

        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for result in results:
            box_num = 0
            
            for i, box in enumerate(result.boxes):
                # Copy the original frame for each bounding box

                bound_image = frame.copy()

                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                probability = box.conf[0].item()
                # Draw only this bounding box on bound_image
                #cv2.rectangle(bound_image, (x1, y1), (x2, y2), (0, 0, 255), 2)


                x_center = x1 + abs(x2 - x1)
                y_center = image_height - (y1 + abs(y2 - y1))
                #print(f"Center: {x_center}, {y_center}")


                # Save the image if display == 1
                #if display == 1:
                #image_path = f"{image_output_dir}/frame_{frame_count}_box_{i+1}.jpg"
                #cv2.imwrite(image_path, bound_image)


                if box_num < len(error) and probability > .40:
                    value, est_long, est_lat =positional_estimate_multi(
                        data[frame_count][2], data[frame_count][3], data[frame_count][5], data[frame_count][4], 
                        data[frame_count][0], data[frame_count][1], x_center, y_center, 
                        true_gps_lat, true_gps_lon
                    )

                    error[box_num].append(1000*value)
                    true_gps_lat_list[box_num].append(true_gps_lat[box_num])
                    true_gps_lon_list[box_num].append(true_gps_lon[box_num])
                    est_lat_list[box_num].append(est_lat)
                    est_lon_list[box_num].append(est_long)


                if probability > .40:
                    box_num += 1   
        end= time.time()
        accumulated_time += end-start
        # Write the original frame to the output video
        out.write(frame)
        tele_count += 1
        frame_count= int(tele_count/3)
    
    average_time.append(accumulated_time/frame_count)

average_time= np.array(average_time)

print("Average time and std per frame:", np.mean(average_time), np.std(average_time)) 


print("Frames processed:", frame_count)
#print("Min error:", np.min(np.array(error)))
#print("Max error:", np.max(np.array(error)))

cap.release()
out.release()
cv2.destroyAllWindows()

"""indices = list(range(len(error[0])))

# Plot bar graph
# Create subplots
fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

colors = ['blue', 'blue', 'blue']
labels = ['Aqua 1 Error Distance', 'Aqua 2 Error Distance', 'Aqua 3 Error Distance']

for i in range(3):
    axes[i].bar(indices, error[i], color=colors[i], alpha=0.7)
    axes[i].set_ylabel("Error Distance (meters)")
    axes[i].set_title(labels[i])
    axes[i].set_ylim(0, 10)
    with open(f"error_multi_2_{i+ 1}.txt", "w") as f:
        for j in range(len(error[i])):
            f.write(f"{error[i][j]}, {true_gps_lat_list[i][j]}, {true_gps_lon_list[i][j]}, {est_lat_list[i][j]}, {est_lon_list[i][j]}\n")


# Set common x-label
axes[-1].set_xlabel("Time Step")
fig.suptitle("Tripple Aqua Localization Sample 2", fontsize=14)
# Adjust layout
plt.tight_layout()
plt.savefig("./plots/mutli_2.png", dpi=300, bbox_inches="tight")  # Saves as PNG with high resolution"""