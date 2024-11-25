import pandas as pd
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# download all required corpus
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data():
    # Load data:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_directory = os.path.join(base_dir, "Youtube05-Shakira.csv")
    return pd.read_csv(file_directory)


def check_basic_data(df: pd.DataFrame):
    # print first 3 rows
    print("\nfirst 3 rows:\n", df.head(3))

    # print the shape of the dataframe
    print("\nshape of the dataframe:\n", df.shape)

    # print the column names
    # print the column types
    # print the missing values per column
    print(df.info())
    print("\nthe missing values per column:", df.isnull().sum(axis=0))

    # print the unique values for all columns
    for cl_name in df.columns:
        print(f"unique values for '{cl_name}' column:", len(df[cl_name].unique()))


def pre_precess(df: pd.DataFrame):
    # drop columns that we don't need. We need to keep only CONTENT and CLASS
    df.drop(
        labels=["COMMENT_ID", "AUTHOR", "DATE"],
        axis=1,
        inplace=True,
        errors="ignore",
    )

    print("\nshape of the dataframe:\n", df.shape)

    # lowercase all content
    df["CONTENT"] = df["CONTENT"].str.lower()

    # remove html tags: TODO not sure whether we need ot do it because some comments have links and it's a good way to detect spam
    remove_html_tags = re.compile(r'<[/]?[a-zA-Z0-9\s]+[/]?>')
    df["CONTENT"] = [re.sub(remove_html_tags, '', c) for c in df["CONTENT"]]

    # replaced escapled characters
    df["CONTENT"] = df["CONTENT"].str.replace('&#39;', '\'')
    df["CONTENT"] = df["CONTENT"].str.replace("&lt;", "<")
    df["CONTENT"] = df["CONTENT"].str.replace("&gt;", ">")

    # tokenize all content
    df["tokens"] = [word_tokenize(c) for c in df["CONTENT"]]

    stop_symbols = [
        '!', '?', '.', ',', '-', '{', '}', '(', ')', '[', ']', '<', '>'
    ]
    en_stopwords = stopwords.words("english") + stop_symbols

    # remove all stop words from content
    df["tokens"] = [[word for word in tokens if word not in en_stopwords] for tokens in df["tokens"]]

    # lemmatize all tokens
    lemmatizer = WordNetLemmatizer()
    df["tokens"] = [[lemmatizer.lemmatize(word) for word in tokens] for tokens in df["tokens"]]

    print("\nfirst 3 rows:\n", df["tokens"].head(10))

    return df


def main():
    comments_df = load_data()
    check_basic_data(comments_df)

    pre_precess(comments_df)


if __name__ == "__main__":
    main()
