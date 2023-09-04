import matplotlib.pyplot as plt
import pandas as pd


def plot_hist(df):
    seq_len = []
    for text in df:
        if isinstance(text, str):
            seq_len.append(len(text.split()))
    pd.Series(seq_len).hist(bins=30)
    # add title and labels
    plt.title('CNN Data')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.show()
