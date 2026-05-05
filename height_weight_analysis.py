import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def hist_of_means(arr, n):
    if len(arr) % n != 0:
        raise ValueError("Array length must be divisible by n.")

    reshaped_arr = arr.reshape(-1, n)
    means = reshaped_arr.mean(axis=1)

    plt.hist(means, bins=15, edgecolor="black")
    plt.xlabel("Mean value")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Means (n={n})")
    plt.show()


def clean_height_weight_data(df):
    df = df.copy()

    df.loc[(df["height_inches"] < 50) | (df["height_inches"] > 85), "height_inches"] = np.nan
    df.loc[(df["weight_pounds"] < 100) | (df["weight_pounds"] > 400), "weight_pounds"] = np.nan

    df["height_inches"] = df["height_inches"].fillna(df["height_inches"].mean())
    df["weight_pounds"] = df["weight_pounds"].fillna(df["weight_pounds"].mean())

    df["height_cm"] = df["height_inches"] * 2.54
    df["weight_kg"] = df["weight_pounds"] * 0.4536

    return df


def plot_histogram(df, column, xlabel, title):
    df[column].plot.hist(bins=15, edgecolor="black")
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.title(title)
    plt.show()


def main():
    array_from_csv = np.loadtxt("data_q1.csv")

    for n in [1, 2, 5, 10, 25, 50]:
        hist_of_means(array_from_csv, n)

    df = pd.read_csv("height_and_weight.csv")

    print("Missing values before cleaning:")
    print(df.isnull().sum())

    df_clean = clean_height_weight_data(df)

    print("\nMissing values after cleaning:")
    print(df_clean.isnull().sum())

    print("\nFirst rows of cleaned data:")
    print(df_clean.head())

    plot_histogram(df_clean, "height_cm", "Height (cm)", "Height Distribution")
    plot_histogram(df_clean, "weight_kg", "Weight (kg)", "Weight Distribution")


if __name__ == "__main__":
    main()
