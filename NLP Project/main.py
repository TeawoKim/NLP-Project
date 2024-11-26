from data_loader import load_data
from data_preprocessor import preprocess_text, process_bow_and_tfidf


if __name__ == "__main__":

    data_files = [
        "data/Youtube01-Psy.csv",
        "data/Youtube02-KatyPerry.csv",
        "data/Youtube03-LMFAO.csv",
        "data/Youtube04-Eminem.csv",
        "data/Youtube05-Shakira.csv",
        ]

    for file_path in data_files:
        print(f"Processing file: {file_path}")
        data = load_data(file_path)
        
        data["Processed_Comment"] = data["Comment"].apply(preprocess_text)

        X_counts, X_tfidf, vectorizer, tfidf_transformer = process_bow_and_tfidf(data)

        feature_names = vectorizer.get_feature_names_out()
        print("Example Features:", feature_names[:10])