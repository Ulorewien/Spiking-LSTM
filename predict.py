import string
import torch
from util import encode, get_all_classes

all_letters = string.ascii_letters + " .,;'"
num_letters = len(all_letters)
all_classes = get_all_classes("data\\names")
num_predictions = 3

model = torch.load("spiking_lstm.pth")

while True:
    name = input("Enter a name: ")
    print(f"-> Name : {name}")
    
    predictions = model(encode(name, num_letters, all_letters))

    top_value, top_class = predictions.topk(num_predictions, 1, True)

    for i in range(num_predictions):
        pred_value = top_value[0][i].item()
        pred_class = top_class[0][i].item()
        print(f"{all_classes[pred_class]} ({pred_value:.2f})")

    cont_pred = input(f"\nDo you want to enter another name? (y/n): ")
    if cont_pred == "n":
        break