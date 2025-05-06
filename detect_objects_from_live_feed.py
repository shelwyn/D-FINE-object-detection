import os
import cv2
import torch
import time
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from transformers import DFineForObjectDetection, AutoImageProcessor

# Set environment variable to disable SSL verification
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = '1'

# Alternative: Use a direct patch for requests library
import requests
requests.packages.urllib3.disable_warnings()
old_merge_environment_settings = requests.Session.merge_environment_settings

def new_merge_environment_settings(self, url, proxies, stream, verify, cert):
    settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
    settings['verify'] = False
    return settings

requests.Session.merge_environment_settings = new_merge_environment_settings

# Function to draw bounding boxes on an image
def draw_boxes(image, results, model_config):
    # Define colors for different object categories (for variety)
    COLORS = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Yellow
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Cyan
        (128, 0, 0),      # Maroon
        (0, 128, 0),      # Dark Green
        (0, 0, 128),      # Navy
        (128, 128, 0),    # Olive
    ]
    
    # Process each detected object
    for result in results:
        for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
            score = score.item()
            label = model_config.id2label[label_id.item()]
            
            # Convert box coordinates to integers for drawing
            x1, y1, x2, y2 = map(int, box.tolist())
            
            # Only draw high-confidence detections
            if score >= 0.5:
                # Choose color based on label_id
                color = COLORS[label_id.item() % len(COLORS)]
                
                # Draw rectangle
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Prepare label text with score
                label_text = f"{label}: {score:.2f}"
                
                # Calculate text size
                (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                
                # Draw background for text
                cv2.rectangle(
                    image, 
                    (x1, y1 - text_height - 5), 
                    (x1 + text_width, y1), 
                    color, 
                    -1
                )
                
                # Draw text
                cv2.putText(
                    image,
                    label_text,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )
    
    return image

def main():
    # Load model and image processor
    print("Loading model... this may take a moment")
    image_processor = AutoImageProcessor.from_pretrained("ustc-community/dfine_x_coco", trust_remote_code=True)
    model = DFineForObjectDetection.from_pretrained("ustc-community/dfine_x_coco", trust_remote_code=True)
    print("Model loaded successfully!")

    # Initialize webcam
    cap = cv2.VideoCapture(0)  # 0 is usually the default camera
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Set frame dimensions (you can adjust these)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # FPS calculation variables
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    print("Press 'q' to quit")
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break
            
        # Convert to RGB (PIL format) for the model
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Process image with model
        inputs = image_processor(images=pil_image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Post-process results
        results = image_processor.post_process_object_detection(
            outputs, 
            target_sizes=[(pil_image.height, pil_image.width)], 
            threshold=0.5
        )
        
        # Draw boxes on OpenCV frame (we're back to BGR now)
        frame = draw_boxes(frame, results, model.config)
        
        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
            
        # Display FPS
        cv2.putText(
            frame, 
            f"FPS: {fps:.2f}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )
        
        # Display the resulting frame
        cv2.imshow('D-FINE Object Detection', frame)
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
