import matplotlib.pyplot as plt
import csv
import numpy as np

FILE_BASE = 'loss_csvs/'

def plot_(x_val, y_val, x_label='Epochs',  y_label='Loss', title='Loss Scores'):
    plt.figure(0)
    ax = plt.subplot()
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    #ax.set_ylim([-1, 1])
    plt.title(title)
    plt.plot(x_val, y_val, marker='o')
    plt.show()



epochs = []
x_val = []

with open(FILE_BASE + 'iic_ave_e99.csv', 'r') as csvfile:
    plots = csv.DictReader(csvfile)
    i = 0
    for row in plots:
        if i % 10 == 0:
            epochs.append(float(row['Col0']))
            x_val.append(int(i))
        i += 1

    epochs = np.asarray(epochs) * -1
plot_(x_val, epochs)


epochs = []
x_val = []
with open(FILE_BASE + 'iic_discrete_e99.csv', 'r') as csvfile:
    plots = csv.DictReader(csvfile)
    i = 0
    for row in plots:
        if i % 10 == 0:
            epochs.append(float(row['Col0']))
            x_val.append(int(i))
        i += 1

    epochs = np.asarray(epochs) * -1
    print(epochs)
plot_(x_val, epochs, x_label='x')


