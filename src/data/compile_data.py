import os
import pandas as pd

from src.utils.reading import read_data

class CompileData:
    '''
    This compiles the data into one csv file for training.
    '''
    def __init__(self , data_path : str , output_path : str):
        self.data_path = data_path
        self.output_path = output_path

    def compile_data(self):
        all_data = os.listdir(self.data_path)
        final_df = pd.DataFrame([])


        for league_data in all_data:
            league_data_path = f'{self.data_path}/{league_data}'
            csv_data = pd.read_csv(league_data_path , index_col=None)
            final_df = pd.concat([final_df , csv_data])

        return final_df


if __name__ == "__main__":
    ...
