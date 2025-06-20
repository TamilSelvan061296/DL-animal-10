import json
import requests
import numpy as np
from PIL import Image

def preprocess_image(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    # ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    for c in range(3):
        arr[..., c] = (arr[..., c] - mean[c]) / std[c]
    return arr.transpose(2, 0, 1)  # CHW

def predict_via_rest(img_tensor: np.ndarray,
                     server_url: str = "http://44.222.155.205:5000/invocations"
                    ) -> np.ndarray:
    payload = {"instances": [img_tensor.tolist()]}
    headers = {"Content-Type": "application/json"}
    resp = requests.post(server_url, headers=headers, data=json.dumps(payload))
    resp.raise_for_status()
    resp_json = resp.json()

    # Extract the list of predictions
    if "predictions" in resp_json:
        preds_list = resp_json["predictions"]
    elif "instances" in resp_json:
        preds_list = resp_json["instances"]
    else:
        raise ValueError(f"Unexpected response format: {resp_json}")

    preds = np.array(preds_list)
    # If it's 1-D or wrapped oddly, ensure shape is (1, num_classes)
    if preds.ndim == 1:
        preds = preds.reshape(1, -1)
    return preds

if __name__ == "__main__":
    IMAGE_PATH = "/mnt/c/Users/TamilSelvanMurugesan/Downloads/random_dog.jpg"
    tensor     = preprocess_image(IMAGE_PATH)
    preds      = predict_via_rest(tensor)

    print("Raw model output:\n", preds)
    class_idxs = preds.argmax(axis=1)
    print("Predicted class indices:", class_idxs)
    print("Single-image prediction:", class_idxs[0])
