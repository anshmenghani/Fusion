# Deletes processed data from the unprocessed folder

import os

f = os.listdir("src/data/formattedData")
u = [x.replace(".gz", "") for x in os.listdir("src/data/unformattedData")]

for i in f:
    if i in u:
        os.remove("src/data/unformattedData/" + i + ".gz")
