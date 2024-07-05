import pandas as pd
import chardet

def read_data(data_path : str):
        # Detect encoding   
        with open(data_path, 'rb') as f:
            result = chardet.detect(f.read())

        # Read file with detected encoding
        df = pd.read_csv(data_path, encoding=result['encoding'])

        return df