import pandas as pd
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import re

# download all required corpus
nltk.download('punkt_tab')
nltk.download('punkt')
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

    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    en_stopwords = stopwords.words("english")
    remove_html_tags = re.compile(r'<[/]?[a-zA-Z0-9\s]+[/]?>')

    def clear_tokens(comment: str):
        # lowercase all content
        l_comment = comment.lower()

        # remove html tags
        cleared_comment = re.sub(remove_html_tags, '', l_comment)

        # replaced escapled characters
        cleared_comment = cleared_comment.replace('&#39;', '\'')
        cleared_comment = cleared_comment.replace("&lt;", "<")
        cleared_comment = cleared_comment.replace("&gt;", ">")

        # convert comments in list of tokens
        tokens = word_tokenize(cleared_comment)

        # filter stopwords, smiley, numbers and punctuation
        tokens = [t for t in tokens if t.isalpha() and t not in en_stopwords]

        # lemmatize all tokens
        lemmas = [lemmatizer.lemmatize(token) for token in tokens]

        # stemming all tokens
        stemmed_tokens = [stemmer.stem(lemma) for lemma in lemmas]

        return ' '.join(stemmed_tokens)

    df['preprocessed_comments'] = df['CONTENT'].apply(clear_tokens)

    print("\nfirst 3 rows:\n", df["preprocessed_comments"].head(15))

    return df


def main():
    comments_df = load_data()
    check_basic_data(comments_df)

    pre_precess(comments_df)


if __name__ == "__main__":
    main()
