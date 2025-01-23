import os
import cv2
import numpy as np

class GCVWorker:
    def __init__(self, width, height):
        os.chdir(os.path.dirname(__file__))
        self.gcvdata = bytearray([0x00])
        self.width = width
        self.height = height
        
        # Create a window for the range slider and set trackbars for the min and max width
        cv2.namedWindow("Range Slider", cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Min Width", "Range Slider", 0, 100, self.nothing)  # Set max to 100
        cv2.createTrackbar("Max Width", "Range Slider", 10, 100, self.nothing)  # Set max to 100
    
    def nothing(self, x):
        pass
    
    def __del__(self):
        del self.gcvdata
    
    def process(self, frame):
        # Convert the image to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define the base HSV values based on the provided pixel samples
        base_hue = 150
        base_saturation = 245  # Average of 249, 237, and 248
        base_value = 241  # Average of 254, 224, and 244
        
        # Increase the tolerance for detecting a broader range of pink hues
        tolerance = 20
        
        # Define the lower and upper bounds for the pink color range
        lower_pink = np.array([base_hue - tolerance, base_saturation - tolerance, base_value - tolerance])
        upper_pink = np.array([base_hue + tolerance, base_saturation + tolerance, base_value + tolerance])
        
        # Create a mask to detect the pink color in the frame
        pink_mask = cv2.inRange(hsv, lower_pink, upper_pink)

        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)  # Kernel for dilation and erosion
        pink_mask = cv2.dilate(pink_mask, kernel, iterations=2)  # Dilation to close small gaps
        pink_mask = cv2.erode(pink_mask, kernel, iterations=1)  # Erosion to remove small noise

        # Find contours of the detected pink area
        contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Initialize detection status and range status
        meter_detected = False
        range_detected = False
        
        # If contours are found, we process the largest contour (assuming it's the dial)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get the bounding box around the contour
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Check if the bounding box area is above a certain threshold (filter out small detections)
            min_area_threshold = 500  # Minimum area to qualify as a valid dial meter detection
            if cv2.contourArea(largest_contour) > min_area_threshold:
                meter_detected = True
            
            # Draw a green rectangle around the pink dial meter
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            # Display the width of the pink dial meter under the green box
            dial_width_text = f"Width: {w}px"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_color = (0, 255, 0)  # Green
            thickness = 2
            cv2.putText(frame, dial_width_text, (x + 10, y + h + 30), font, font_scale, font_color, thickness)
            
            # Get the min and max width from the range slider
            min_width = cv2.getTrackbarPos("Min Width", "Range Slider")
            max_width = cv2.getTrackbarPos("Max Width", "Range Slider")
            
            # Check if the detected width is within the selected range
            if min_width <= w <= max_width:
                range_detected = True
        
        # Add text at the top middle of the image
        text = "PLUGGEDVISION"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (0, 255, 0)  # Green
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (self.width - text_size[0]) // 2
        text_y = 30  # Position the text at the top middle
        
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, thickness)
        
        # Display the "METER DETECTED" status at the bottom of the image
        detection_status = f"METER DETECTED: {str(meter_detected)}"
        cv2.putText(frame, detection_status, (10, self.height - 60), font, 0.8, font_color, thickness)
        
        # Display the "RANGE DETECTED" status at the bottom of the image
        range_status = f"RANGE DETECTED: {str(range_detected)}"
        cv2.putText(frame, range_status, (10, self.height - 30), font, 0.8, font_color, thickness)

        # Show the updated frame inside the "Range Slider" window
        cv2.imshow("Range Slider", frame)

        # Wait for key press to update the window
        cv2.waitKey(1)

        return frame, None
