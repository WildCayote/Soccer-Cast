import os , chardet , argparse
import pandas as pd

# ignore warnings
import warnings
warnings.simplefilter('ignore')

def read_data(data_path : str):
        # Detect encoding   
        with open(data_path, 'rb') as f:
            result = chardet.detect(f.read())

        # Read file with detected encoding
        df = pd.read_csv(data_path, encoding=result['encoding'])

        return df



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
    args = argparse.ArgumentParser()
    args.add_argument('--data_path' , default='./data/interim/compiled')
    args.add_argument('--export_path' , default='./data/processed')
    
    parsed_args = args.parse_args()

    DATA_PATH = parsed_args.data_path
    OUTPUT_PATH = parsed_args.export_path

    all_data = os.listdir(DATA_PATH)
    final_df = pd.DataFrame([])


    for league_data in all_data:
        league_data_path = f'{DATA_PATH}/{league_data}'
        try: 
            csv_data = pd.read_csv(league_data_path , index_col=None)
            final_df = pd.concat([final_df , csv_data])
        except Exception as e:
            print(league_data_path)

    # remove the index column
    final_df = final_df.drop(columns=final_df.columns[0])

    # write the data
    save_path = f'{OUTPUT_PATH}/final.csv'
    final_df.to_csv(save_path , index=False)