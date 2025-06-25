from test_serve import preprocess_image, predict_via_rest

from dl_animal_10.data.data_etl import load_and_transform, verify_the_dataset
from dl_animal_10.config.config_loader import Config
import torch

cfg = Config('/home/tamil/DL-animal-10/src/dl_animal_10/config/config.yaml')

train, test = load_and_transform(cfg)
data, target = next(iter(test))
accuracy = 0
correct_predictions = []
wrong_predictions = []
correct_counts = {}
wrong_counts = {}
steps = 0
for inputs, labels in test:
    steps += 1
    for i in range(len(inputs)):
        preds      = predict_via_rest(inputs[i])

        # print("Raw model output:\n", preds)
        class_idxs = preds.argmax(axis=1)
        # print("Predicted class indices:", class_idxs)
        # print("Single-image prediction:", class_idxs[0])
        # print("Actual class:", labels[i].item())
        if class_idxs[0] != labels[i].item():
            wrong_predictions.append(class_idxs[0])
        if class_idxs[0] == labels[i].item():
            correct_predictions.append(class_idxs[0])
            accuracy += 1
    print(accuracy/(steps*len(inputs)))
    if steps % 20 == 0:
        break

print("final accuracy:", accuracy/len(test))

for item in correct_predictions:
    correct_counts[item] = correct_counts.get(item, 0) + 1
print(correct_counts)

for item in wrong_predictions:
    wrong_counts[item] = wrong_counts.get(item, 0) + 1
print(wrong_counts)
print("done")