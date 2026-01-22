from flask import Flask, request
import pandas as pd
import numpy as np

app = Flask(__name__)

def topsis(df, weights, impacts):
    data = df.iloc[:, 1:]

    norm_data = data / np.sqrt((data ** 2).sum())
    weighted_data = norm_data * weights

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

    dist_best = np.sqrt(((weighted_data - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted_data - ideal_worst) ** 2).sum(axis=1))

    score = dist_worst / (dist_best + dist_worst)

    df["Topsis Score"] = score
    df["Rank"] = df["Topsis Score"].rank(ascending=False).astype(int)

    return df

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files.get("file")
        weights = request.form.get("weights")
        impacts = request.form.get("impacts")

        if not file or not weights or not impacts:
            return "Please upload file and enter weights & impacts"

        df = pd.read_csv(file)
        weights = list(map(float, weights.split(",")))
        impacts = impacts.split(",")

        result = topsis(df, weights, impacts)
        return result.to_html(index=False)

    return '''
    <h2>TOPSIS Web App</h2>
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="file"><br><br>
        <input type="text" name="weights" placeholder="Weights (e.g. 1,1,1,1,1)"><br><br>
        <input type="text" name="impacts" placeholder="Impacts (e.g. +,+,-,-,-)"><br><br>
        <input type="submit" value="Calculate TOPSIS">
    </form>
    '''

if __name__ == "__main__":
    app.run(debug=True)
