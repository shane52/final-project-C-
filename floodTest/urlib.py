import cv2
import numpy as np
import csv
import datetime
import threading
import queue
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
from collections import deque
import imutils

# Initialize the video capture in a separate thread
def video_capture_thread():
    global cap
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        if ret:
            frame_queue.put(img)
        else:
            print("Error capturing video frame. Retrying...")
        time.sleep(0.1)  # Adjust the sleep time to control the frame rate

# Create a queue for storing video frames
frame_queue = queue.Queue()

# Create a CSV file to store the water level data
csv_file_path = r'D:\Flood detection folder\water_level_data.csv'
csv_file = open(csv_file_path, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Timestamp', 'Water Level'])

# Initialize the water level data lists for visualization
water_levels = []
timestamps = []

threshold = 200  # Set the threshold value for water overflow

# Start the video capture thread
video_thread = threading.Thread(target=video_capture_thread)
video_thread.daemon = True
video_thread.start()

# Function to update the graph
def update_graph():
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    while True:
        if timestamps:
            ax.clear()
            ax.plot(timestamps, water_levels, marker='o')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            fig.autofmt_xdate()
            ax.set_xlabel('Timestamp')
            ax.set_ylabel('Water Level')
            ax.set_title('Water Level Over Time')
            ax.grid(True)
            plt.draw()
            plt.pause(1)

# Start the graph updating thread
graph_thread = threading.Thread(target=update_graph)
graph_thread.daemon = True
graph_thread.start()

try:
    while True:
        if not frame_queue.empty():
            img = frame_queue.get()
            img1 = img.copy()
            img = img[5:600, 0:43]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            ret, thresh1 = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            drawing = np.zeros(img.shape, np.uint8)
            drawing = drawing[20:500, 22:24]
            max_area = 0
            ci = -1
            for i in range(len(contours)):
                cnt = contours[i]
                area = cv2.contourArea(cnt)
                if area > max_area:
                    max_area = area
                    ci = i
            if ci != -1:
                cnt = contours[ci]
                hull = cv2.convexHull(cnt)
                moments = cv2.moments(cnt)
                if moments['m00'] != 0:
                    cx = int(moments['m10'] / moments['m00'])
                    cy = int(moments['m01'] / moments['m00'])
                    centr = (cx, cy)
                    cv2.circle(drawing, centr, 1, [0, 0, 255], 2)
                    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 2)
                    cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 2)

                    # Resize the frame for better performance
                    drawing = cv2.resize(drawing, (320, 240))

                    # Check if the water level exceeds the threshold
                    if centr[1] > threshold:
                        print("Water Overflow detected!")
                        cv2.putText(img1, "Water Overflow!", (40, centr[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA, False)
                        cv2.rectangle(img1, (40, centr[1]), (120, centr[1] + 20), (0, 0, 255), -1)

                    # Store the water level data in the CSV file
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    water_level = centr[1]
                    csv_writer.writerow([timestamp, water_level])
                    water_levels.append(water_level)
                    timestamps.append(datetime.datetime.now())

            # Display the original and processed video feeds
            cv2.imshow('Original Feed', img1)
            cv2.imshow('Processed Feed', drawing)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
finally:
    # Release the video capture, close the CSV file, and close the visualization window
    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()

# Display the water level graph
plt.ioff()
plt.show()
