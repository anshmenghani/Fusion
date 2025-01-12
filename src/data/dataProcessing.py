# Unpacks .csv.gz files and extracts usable training, validation, and testing data out of them

from astropy.io import ascii
import pandas as pd
import os
import shutil

drop_list = []
def checker(value):
    for i in value:
        try:
            if i == "null" and value.name != drop_list[-1]:
                drop_list.append(value.name)
        except IndexError:
            drop_list.append(value.name)
    return value

main_dir = "src/data/unformattedData/"
for file in os.listdir(main_dir):
    path = main_dir + file
    print(path)
    table = ascii.read(path)
    df = table.to_pandas()
    df = df[['teff_gspphot_phoenix', 'mg_gspphot_phoenix', 'logg_gspphot_phoenix', 'radius_gspphot_phoenix', 
              'mh_gspphot_phoenix']]
    
    df = df.apply(checker, axis=1)
    range_list = range(df.shape[0])
    uvals = set(drop_list) ^ set(range_list)
    new_df = pd.DataFrame(columns=['teff_gspphot_phoenix', 'mg_gspphot_phoenix', 'logg_gspphot_phoenix', 'radius_gspphot_phoenix', 
              'mh_gspphot_phoenix'])
    
    for idx, val in enumerate(uvals):
        try:
            new_df.loc[idx] = df.iloc[val]
        except IndexError:
            continue
    new_df.to_csv(path.replace(".gz", ""))
    shutil.move(path.replace(".gz", ""), "src/data/formattedData")
