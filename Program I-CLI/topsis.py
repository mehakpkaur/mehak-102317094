import sys
import pandas as pd
import numpy as np
import os

def topsis(input_file, weights, impacts, output_file):

    # ---------- File check ----------
    if not os.path.exists(input_file):
        print("Error: Input file not found")
        sys.exit(1)

    # ---------- Read input file ----------
    if input_file.endswith(".csv"):
        df = pd.read_csv(input_file)
    elif input_file.endswith(".xlsx"):
        df = pd.read_excel(input_file)
    else:
        print("Error: Input file must be a CSV or XLSX file")
        sys.exit(1)

    # ---------- Column check ----------
    if df.shape[1] < 3:
        print("Error: Input file must contain three or more columns")
        sys.exit(1)

    data = df.iloc[:, 1:]

    # ---------- Numeric check ----------
    if not np.all(data.applymap(np.isreal)):
        print("Error: From 2nd column to last column must contain numeric values only")
        sys.exit(1)

    weights = list(map(float, weights.split(",")))
    impacts = impacts.split(",")

    # ---------- Length check ----------
    if len(weights) != data.shape[1] or len(impacts) != data.shape[1]:
        print("Error: Number of weights, impacts and columns must be same")
        sys.exit(1)

    # ---------- Impact check ----------
    for i in impacts:
        if i not in ['+', '-']:
            print("Error: Impacts must be either + or -")
            sys.exit(1)

    # ---------- Normalization ----------
    norm_data = data / np.sqrt((data ** 2).sum())

    # ---------- Weighting ----------
    weighted_data = norm_data * weights

    # ---------- Ideal best & worst ----------
    ideal_best = []
    ideal_worst = []

    for i in range(len(impacts)):
        if impacts[i] == '+':
            ideal_best.append(weighted_data.iloc[:, i].max())
            ideal_worst.append(weighted_data.iloc[:, i].min())
        else:
            ideal_best.append(weighted_data.iloc[:, i].min())
            ideal_worst.append(weighted_data.iloc[:, i].max())

    ideal_best = np.array(ideal_best)
    ideal_worst = np.array(ideal_worst)

    # ---------- Distance ----------
    dist_best = np.sqrt(((weighted_data - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted_data - ideal_worst) ** 2).sum(axis=1))

    # ---------- Topsis score ----------
    score = dist_worst / (dist_best + dist_worst)

    df['Topsis Score'] = score
    df['Rank'] = df['Topsis Score'].rank(ascending=False).astype(int)

    df.to_csv(output_file, index=False)
    print("Topsis calculation completed successfully")


def main():
    if len(sys.argv) != 5:
        print("Usage: topsis <InputFile> <Weights> <Impacts> <OutputFile>")
        sys.exit(1)

    _, input_file, weights, impacts, output_file = sys.argv
    topsis(input_file, weights, impacts, output_file)
