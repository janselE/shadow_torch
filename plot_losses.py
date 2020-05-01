import matplotlib.pyplot as plt
import csv
import numpy as np

FILE_BASE = 'loss_csvs/'

def plot_(x_val, y_val1, y_val2=None, x_label='Epochs',  y_label='Loss', title='Loss Scores'):
    plt.figure(0)
    ax = plt.subplot()
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    #ax.set_ylim([-1, 1])
    plt.title(title)
    plt.plot(x_val, y_val1, marker='o')
    plt.plot(x_val, y_val2, marker='o')
    plt.show()

def reader(file_path):
    y_val1 = []
    y_val2 = []
    x_val = []
    with open(file_path, 'r') as csvfile:
        plots = csv.DictReader(csvfile)
        i = 0
        for row in plots:
            if i % 10 == 0:
                y_val1.append(float(row['Col0']))
                y_val2.append(float(row['Col1']))
                x_val.append(int(i))
            i += 1

        y_val1 = np.asarray(y_val1) * 1
        y_val2 = np.asarray(y_val2) * 1
    plot_(x_val, y_val1, y_val2)


reader(FILE_BASE + 'iic_ave_e99.csv')
reader(FILE_BASE + 'iic_discrete_e99.csv')
