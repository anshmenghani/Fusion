# Transform CSV to split into two and use GZIP compression (.csv -> .csv.gz extension)

import pandas as pd
from math import floor

df = pd.read_csv("src/FUSION/FusionStellaarData.csv")
half = floor(df.shape[0]/2)
df1 = df.iloc[:half, :]
df2 = df.iloc[half:, :]
df1.to_csv("src/FUSION/FusionStellaarData1.csv.gz", compression="gzip")
df2.to_csv("src/FUSION/FusionStellaarData2.csv.gz", compression="gzip")
