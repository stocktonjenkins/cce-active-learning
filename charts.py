import os
import matplotlib.pyplot as plt
import pandas as pd

root = "/Users/stocktonjenkins/Downloads/cce-logs-post-midterm"
names = ["badge", "cce", "confidence", "random", "reverse_cce"]
# names = ["badge", "cce", "confidence", "reverse_cce"]

line_styles = {"ECE": "-", "Mean MCMD": "--"}  # Line styles for MAE and RMSE
colors = [
    "#EF4236",
    "#3352EF",
    "#DBBD26",
    "#B7B167",
    "#71BFCA",
    "#E1C866",
    "#17EDE8",
    "#3B030F",
    "#2C34D7",
    "#30F2B1",
]

for i, prefix in enumerate(names):
    df = pd.read_csv(
        os.path.join(root, f"{prefix}__aaf_gaussian/al_model_acc.csv"),
    ).drop_duplicates()
    mae_diff = abs(df.iloc[2]["MAE"] - df.iloc[0]["MAE"])
    rmse_diff = abs(df.iloc[2]["RMSE"] - df.iloc[0]["RMSE"])
    mae__delta_diff = abs((df.iloc[2]["MAE"] - df.iloc[0]["MAE"]) / df.iloc[0]["MAE"])
    rmse__delta_diff = abs((df.iloc[2]["RMSE"] - df.iloc[0]["RMSE"]) / df.iloc[0]["RMSE"])
    print({
        "name": prefix,
        "mae": df.iloc[2]["MAE"],
        "rmse": df.iloc[2]["RMSE"],
        "mae_diff": mae_diff,
        "rmse_diff": rmse_diff,
        "mae__delta_diff": mae__delta_diff,
        "rmse__delta_diff": rmse__delta_diff,
    })


# dataframes = {}
# for i, prefix in enumerate(names):
#     df = pd.read_csv(
#         os.path.join(root, f"{prefix}__aaf_gaussian/al_model_calibration.csv"),
#     ).drop_duplicates()
#     dataframes[prefix] = df
#
#
# plt.figure(figsize=(10, 6))
# for i, values in enumerate(dataframes.items()):
#     label, df = values
#     for col in ["ECE", "Mean MCMD"]:
#         plt.plot(
#             df["Training Set Size"],
#             df[col],
#             color=colors[i],
#             label=f"{label} - {col}",
#             linestyle=line_styles[col],
#         )
#
# # Customize the plot
# plt.xlabel("Label Set Size")
# plt.ylabel("Probabilistic Fit")
# plt.legend()
# plt.grid(True)
#
# # Show the plot
# plt.tight_layout()
# plt.savefig("CAL.png")
# plt.close()
