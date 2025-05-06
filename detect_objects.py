import os
import torch
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

# Now try loading the model
url = 'https://shelwyn.in/images/IM2A.jpeg'
image = load_image(url)

image_processor = AutoImageProcessor.from_pretrained("ustc-community/dfine_x_coco", trust_remote_code=True)
model = DFineForObjectDetection.from_pretrained("ustc-community/dfine_x_coco", trust_remote_code=True)

inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

results = image_processor.post_process_object_detection(outputs, target_sizes=[(image.height, image.width)], threshold=0.5)

for result in results:
    for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
        score, label = score.item(), label_id.item()
        box = [round(i, 2) for i in box.tolist()]
        print(f"{model.config.id2label[label]}: {score:.2f} {box}")
