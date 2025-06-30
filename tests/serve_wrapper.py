import mlflow.pyfunc
import mlflow.pytorch
import torch
from torchvision import transforms
from PIL import Image
import io


class PreprocessingWrapper(mlflow.pyfunc.PythonModel):
    
    
    def load_context(self, context):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = mlflow.pytorch.load_model(context.artifacts["model_dir"], 
                                               map_location=self.device)
        self.model.eval()

        # pre-processing logic
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
    
    def predict(self, context, model_input):
        
        if isinstance(model_input, (bytes, bytearray)):
            imgs = [model_input]
        else:
            imgs = list(model_input)
        
        tensors = []
        for img_bytes in imgs:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            tensors.append(self.transform(img))
        batch = torch.stack(tensors)

        with torch.no_grad():
            logits = self.model(batch)
            preds = torch.argmax(logits, dim = 1).cpu().numpy().tolist()
        
        return preds[0] if len(preds) == 1 else preds



conda_env = {
    "name": "mlflow-env",
    "channels": ["conda-forge"],
    "dependencies": [
        "python=3.12.3",
        "pip",
        {
        "pip":[
            "mlflow==3.1.0",
            "cloudpickle==3.1.1",
            "defusedxml==0.7.1",
            "numpy==2.3.0",
            "pandas==2.3.0",
            "torch==2.7.1",
            "torchvision==0.22.1",
            "tqdm==4.67.1"
        ]
            }
    ]
}


mlflow.pyfunc.save_model(
    path="/home/tamil/DL-animal-10/src/dl_animal_10/models/animal_classifier_with_preprocessing_step",
    python_model=PreprocessingWrapper(),
    artifacts={"model_dir": "/home/tamil/DL-animal-10/src/dl_animal_10/models/animal_classifier_892"},
    conda_env=conda_env
)