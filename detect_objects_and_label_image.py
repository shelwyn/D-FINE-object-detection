import os
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers.image_utils import load_image
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
    # Convert PIL Image to a format we can draw on
    draw = ImageDraw.Draw(image)
    
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
    
    # Try to load a font, falling back to default if not available
    try:
        # Try to use a TrueType font if available
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        # Fall back to default font
        font = ImageFont.load_default()
    
    # Process each detected object
    for result in results:
        for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
            score = score.item()
            label = model_config.id2label[label_id.item()]
            box = [round(i) for i in box.tolist()]
            
            # Only draw high-confidence detections
            if score >= 0.5:
                # Choose color based on label_id
                color = COLORS[label_id.item() % len(COLORS)]
                
                # Draw rectangle
                draw.rectangle([box[0], box[1], box[2], box[3]], outline=color, width=3)
                
                # Prepare label text with score
                label_text = f"{label}: {score:.2f}"
                
                # Get text size (this works with both newer and older versions of PIL)
                if hasattr(font, "getbbox"):
                    # For newer PIL versions
                    text_size = font.getbbox(label_text)
                    text_width = text_size[2] - text_size[0]
                    text_height = text_size[3] - text_size[1]
                else:
                    # For older PIL versions
                    text_width, text_height = draw.textsize(label_text, font=font)
                
                # Draw background for text
                draw.rectangle(
                    [box[0], box[1]-text_height-4, box[0]+text_width+4, box[1]],
                    fill=color
                )
                
                # Draw text (handle different PIL versions)
                if hasattr(draw, "text"):
                    draw.text((box[0]+2, box[1]-text_height-2), label_text, fill="white", font=font)
                else:
                    # Fallback for very old PIL versions
                    draw.text((box[0]+2, box[1]-text_height-2), label_text, fill="white")
    
    return image

# Main code
url = 'https://shelwyn.in/images/IM2A.jpeg'
image = load_image(url)
original_image = image.copy()  # Keep a copy of the original image

# Load model and processor
image_processor = AutoImageProcessor.from_pretrained("ustc-community/dfine_x_coco", trust_remote_code=True)
model = DFineForObjectDetection.from_pretrained("ustc-community/dfine_x_coco", trust_remote_code=True)

# Process the image
inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

results = image_processor.post_process_object_detection(outputs, target_sizes=[(image.height, image.width)], threshold=0.5)

# Print detection results
for result in results:
    for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
        score, label = score.item(), label_id.item()
        box = [round(i, 2) for i in box.tolist()]
        print(f"{model.config.id2label[label]}: {score:.2f} {box}")

# Draw bounding boxes on the image
annotated_image = draw_boxes(original_image, results, model.config)

# Save the annotated image
output_path = "annotated_image.jpg"
annotated_image.save(output_path)
print(f"Annotated image saved to {output_path}")
