import cv2
import numpy as np
import time
import csv

# Input video file
cam = cv2.VideoCapture(r"C:\Users\manju\OneDrive\Desktop\python\vedi.mp4")

bounce_count = 0
prev_center_y = None
upward = False
start_time = time.time()
total_bounce_time = 0  # Initialize total bounce time

# Create and open a CSV file to write data
csv_file = open('bounce_data.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Count', 'Time'])
print('Count, Time')

while True:
    _, frame = cam.read()
    if frame is None:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian Blur to the grayscale frame
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # Use the Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=50, param2=30, minRadius=10,
                               maxRadius=100)

    # If circles are detected, draw a rectangle around the first one and count bounces
    if circles is not None:
        circles = np.uint16(np.around(circles))
        circle = circles[0, 0]  # Select the first detected circle
        x, y, radius = circle
        x, y, radius = int(x), int(y), int(radius)  # Convert to integers
        cv2.rectangle(frame, (x - radius, y - radius), (x + radius, y + radius), (0, 255, 0), 2)

        # Check if the ball is moving downwards
        if prev_center_y is not None:
            if y > prev_center_y:
                upward = False
            elif y < prev_center_y and not upward:
                bounce_count += 1
                upward = True
                # Accumulate the total time of bounces
                current_time = time.time()
                total_bounce_time += current_time - start_time
                start_time = current_time

                # Print data in terminal and write to CSV
                data = f'{bounce_count}, {total_bounce_time:.2f}'
                print(data)
                csv_writer.writerow([bounce_count, total_bounce_time])

        # Display the total bounce time on the top-right corner
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f'Count: {bounce_count}', (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Time: {total_bounce_time:.2f} s', (frame.shape[1] - 250, 30), font, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)

        prev_center_y = y

    cv2.imshow('Ball Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit the camera
        break

# Release the video object and close the windows
cam.release()
cv2.destroyAllWindows()

# Close the CSV file
csv_file.close()