import unicodedata
import torch
import time
import matplotlib.pyplot as plt

def unicodeToAscii(s, all_letters):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def getIndex(letter, all_letters):
    return all_letters.find(letter)

def encode(word, num_letters, all_letters, batch_size=1):
    encoded_word = torch.zeros(len(word), batch_size, num_letters)

    # Modify this for loop if you have a batch_size of more than 1
    for i, letter in enumerate(word):
        encoded_word[i][0][getIndex(letter, all_letters)] = 1

    return encoded_word

def predictClass(output, classes):
    top_n, top_i = output.topk(1)
    class_i = top_i[0].item()

    return classes[class_i], class_i

def trainTime(start):
    runtime = time.time() - start
    mins = runtime//60
    runtime -= mins*60
    return f"({mins}m {runtime:.2f}s)"

def plotMetrics(avg_losses, train_acc, test_acc, plot_every):
    plt.figure()

    plt.subplot(311)
    plt.plot(avg_losses)
    plt.title("Average Loss")

    plt.subplot(312)
    plt.plot(train_acc)
    plt.title("Train Accuracy")

    plt.subplot(313)
    plt.plot(test_acc)
    plt.title("Test Accuracy")

    plt.xlabel(f"No. of Epochs (x{plot_every})")
    plt.subplots_adjust(hspace=0.6)
    plt.savefig("SpikingLSTMTraining.svg")
    plt.close()