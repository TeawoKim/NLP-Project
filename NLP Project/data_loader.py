import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path)[["CONTENT", "CLASS"]]
    data.columns = ["Comment", "Label"]
    return data