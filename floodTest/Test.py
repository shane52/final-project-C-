import cv2
import numpy as np
import csv
import datetime
import threading
import queue
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
from statsmodels.tsa.arima.model import ARIMA
from pushbullet import Pushbullet

# Initialize the Pushbullet API with your access token
pb = Pushbullet("o.VSlIrp8EINKxTd7JqVOK97mKje6FFiy3")

# Initialize the video capture in a separate thread
def video_capture_thread(frame_queue):
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

# Function to send Pushbullet notification
def send_notification(title, body):
    pb.push_note(title, body)

# Function to perform time series analysis and make predictions
def predict_water_levels(water_levels):
    while True:
        if len(water_levels) > 5:
            # Perform time series analysis and make predictions using ARIMA modeling
            model = ARIMA(water_levels, order=(1, 1, 1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=5)

            # Display the forecasted water levels
            print("Forecasted Water Levels:")
            print(forecast)

            # Check for potential flood risks based on the forecasted water levels
            for level in forecast:
                if level > threshold:
                    print("Potential flood risk detected! Water level exceeds threshold.")
                    send_notification("Alert", "Potential flood risk detected! Water level exceeds threshold.")
        time.sleep(60)  # Adjust the sleep time based on your requirement

# Initialize the water level data lists for visualization
timestamps = []
threshold = 200

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
                        # Send notification
                        send_notification("Alert", "Water Overflow detected!")
                    else:
                        cv2.putText(img1, "Water Not Overflowing", (40, centr[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA, False)
                        cv2.rectangle(img1, (40, centr[1]), (160, centr[1] + 20), (0, 255, 0), -1)

                    # Store the water level data
                    timestamp = datetime.datetime.now()
                    water_level = centr[1]
                    timestamps.append(timestamp)

            # Display the original and processed video feeds
            cv2.imshow('Original Feed', img1)
            cv2.imshow('Processed Feed', drawing)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
except KeyboardInterrupt:
    print("Keyboard interrupt detected. Exiting...")
finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    # Perform Matplotlib operations in the main thread
    # Initialize the water level data list for visualization
    water_levels = []
    
    # Start the prediction thread
    prediction_thread = threading.Thread(target=predict_water_levels, args=(water_levels,))
    prediction_thread.daemon = True
    prediction_thread.start()

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

    # Wait for all threads to finish before exiting
    prediction_thread.join()
    graph_thread.join()
