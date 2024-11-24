from data_loader import load_data

if __name__ == "__main__":

    data_files = [
        "data/Youtube01-Psy.csv",
        "data/Youtube01-KatyPerry.csv",
        "data/Youtube01-LMFAO.csv",
        "data/Youtube01-Eminem.csv",
        "data/Youtube01-Shakria.csv",
        ]

    for file_path in data_files:
        print(f"Processing file: {file_path}")
        data = load_data(file_path)

       