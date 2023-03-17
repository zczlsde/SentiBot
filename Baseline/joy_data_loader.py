import pandas as pd
from datasets import load_dataset
import os
"""
Used for loading positive data to file
"""

if __name__ == "__main__":
    # ozziey data
    ozziey_path = "./Data/joy_data"
    if not os.path.exists(ozziey_path):
        dataset = load_dataset(
            "Ozziey/poems_dataset", 
            data_files="final_df_emotions(remove-bias).csv"
        )['train']
        joy_dataset = dataset.filter(lambda row: row['label'] == 'joy')
        joy_dataset.save_to_disk(ozziey_path)

    #PERC data
    PERC_path = "./Data/PERC.csv"
    PERC_xlsx_path = "./Data/PERC_mendelly.xlsx"
    overwrite = False
    
    df = pd.DataFrame(pd.read_excel(PERC_xlsx_path))
    print(set(df['Emotion']))
    if not os.path.exists(PERC_path) or overwrite:
        if os.path.exists(PERC_xlsx_path):
            read_file = pd.read_excel(PERC_xlsx_path)
        
            # Write the dataframe object
            # into csv file
            read_file.to_csv (PERC_path, 
                            index = None,
                            header=True)
        else:
            raise ValueError("Incorrect path for xlsx")
