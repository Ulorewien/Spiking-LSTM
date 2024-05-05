import string
import time
from random import randint
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import NamesDataset
from util import encode, predictClass, trainTime, plotMetrics
from model import SpikingLSTM

# Global variables
dataset_dir = "data\\names"
all_letters = string.ascii_letters + " .,;'"
num_letters = len(all_letters)

# Hyperparameters
batch_size = 1
learning_rate = 1e-4
num_epochs = 100000
split_ratio = 0.8
hidden_size = 256

# Additional variables
num_classes = 18
plot_every = 1000
print_every = 100

# Load the dataset
dataset = NamesDataset(root_dir=dataset_dir, all_letters=all_letters, split_ratio=split_ratio)
# train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
test_data = dataset.get_test_data()

# Initialize the model
model = SpikingLSTM(num_letters, hidden_size, num_classes)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

current_loss = 0
correct_samples = 0
avg_losses = []
train_acc = []
test_acc = []
start_time = time.time()

for epoch in range(1, num_epochs+1):
    model.train()

    # for name, lang in train_loader:
    random_idx = randint(0, len(dataset)-1)
    name, lang = dataset[random_idx]
        
    name_tensor = encode(name, num_letters, all_letters)
    lang_tensor = torch.tensor([dataset.languages.index(lang)], dtype=torch.float)
    label_one_hot = F.one_hot(lang_tensor.to(int), num_classes).float()

    pred_label = model(name_tensor)
    loss = F.mse_loss(pred_label, label_one_hot)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    current_loss += loss.data.item()
    pred_class, _ = predictClass(pred_label.data, dataset.languages)
    if pred_class == lang:
        correct_samples += 1
    
    if epoch%plot_every == 0:
        avg_losses.append(current_loss/plot_every)
        train_acc.append(correct_samples/plot_every)
        current_loss = 0
        correct_samples = 0

        model.eval()
        with torch.no_grad():
            num_correct = 0

            for i in range(len(test_data)):
                name, lang = test_data[i]
                pred_label = model(encode(name, num_letters, all_letters))
                pred_class, _ = predictClass(pred_label.data, dataset.languages)
                if pred_class == lang:
                    num_correct += 1
            
            test_acc.append(num_correct/len(test_data))
            print(f"Epoch {epoch} {trainTime(start_time)}: train loss {avg_losses[-1]}; train accuracy {train_acc[-1]}; test accuracy {test_acc[-1]}")

# Save the model and metrics
torch.save(model, "spiking_lstm.pth")
np.save("avg_losses.npy", np.array(avg_losses))
np.save("train_acc.npy", np.array(train_acc))
np.save("test_acc.npy", np.array(test_acc))

# Plot Metrics
plotMetrics(avg_losses, train_acc, test_acc, plot_every)