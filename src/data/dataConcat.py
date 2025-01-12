# Merge individual CSV data files into one dataset 

import pandas as pd
import os

dfs = []
dir_name = "src/data/formattedData"
for f in os.listdir(dir_name):
    path = os.path.join(dir_name, f)
    df = pd.read_csv(path)
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)
combined_df = combined_df.drop_duplicates()
combined_df.reset_index(drop=True, inplace=True)

name = "total_combined_data"
while name in [os.path.basename(x) for x in os.listdir()]:
    name += "_"

combined_df.to_csv(name + ".csv", index=False)
