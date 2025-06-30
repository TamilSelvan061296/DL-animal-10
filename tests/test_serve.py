import json
import requests
import numpy as np
from PIL import Image
import base64

# def preprocess_image(image_path: str) -> np.ndarray:
#     img = Image.open(image_path).convert("RGB")
#     img = img.resize((224, 224))
#     arr = np.array(img, dtype=np.float32) / 255.0
#     # ImageNet normalization
#     # mean = [0.485, 0.456, 0.406]
#     # std  = [0.229, 0.224, 0.225]
#     # for c in range(3):
#     #     arr[..., c] = (arr[..., c] - mean[c]) / std[c]
#     return arr.transpose(2, 0, 1)  # CHW

def predict_via_rest(encoded_image,
                     server_url: str = "http://0.0.0.0:8000/invocations"
                    ) -> np.ndarray:
    headers={"Content-Type": "application/octet-stream"}
    resp = requests.post(server_url, headers=headers, data=encoded_image)
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
    with open("/home/tamil/DL-animal-10/data1/spider/e03db4072cf01c22d2524518b7444f92e37fe5d404b0144390f8c47ba6edbc_640.jpg", "rb") as img_file:
        encoded_image = img_file.read()
    
    
    preds      = predict_via_rest(encoded_image)

    print("Raw model output:\n", preds)
    # class_idxs = preds.argmax(axis=1)
    # print("Predicted class indices:", class_idxs)
    # print("Single-image prediction:", class_idxs[0])
