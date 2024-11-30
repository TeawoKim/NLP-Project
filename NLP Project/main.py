from data_loader import load_data
from data_preprocessor import preprocess_text, process_bow_and_tfidf
from evaluator import train_and_evaluate_model, stratified_split_data


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
        
        x_tfidf, tfidf_vectorizer = process_bow_and_tfidf(data)
        
        x_train, x_test, y_train, y_test = stratified_split_data(data, x_tfidf, train_ratio=0.75)
        
        train_and_evaluate_model(x_train, x_test, y_train, y_test, tfidf_vectorizer)