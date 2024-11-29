import pandas as pd
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

    def clear_comments(comment: str):
        # lowercase all content
        l_comment = comment.lower()

        # remove html tags
        cleared_comment = re.sub(remove_html_tags, '', l_comment)

        # replaced escapled characters
        cleared_comment = cleared_comment.replace('&#39;', '\'')
        cleared_comment = cleared_comment.replace("&lt;", "<")
        cleared_comment = cleared_comment.replace("&gt;", ">")

        # remove unknown symbol at the end of each coment
        cleared_comment = cleared_comment.replace("\ufeff", "")

        # convert comments in list of tokens
        tokens = word_tokenize(cleared_comment)

        # filter stopwords, smiley, numbers and punctuation
        tokens = [t for t in tokens if t.isalpha() and t not in en_stopwords]

        # lemmatize all tokens
        lemmas = [lemmatizer.lemmatize(token) for token in tokens]

        # stemming all tokens
        stemmed_tokens = [stemmer.stem(lemma) for lemma in lemmas]

        return ' '.join(stemmed_tokens)

    # iterate over each comment, clear it and convert it to list of tokens
    df['preprocessed_comments'] = df['CONTENT'].apply(clear_comments)

    print("\nfirst 5 rows:\n", df["preprocessed_comments"].head(5))

    return df


def vectorize_tokens(df: pd.DataFrame):
    documents = df["preprocessed_comments"]

    # vectorize text with bag of words
    bw_vectorizer = CountVectorizer()
    bw_vectorized = bw_vectorizer.fit_transform(documents)

    print("\nshape of the vectorized dataframe:\n", bw_vectorized.shape)
    bw_features = bw_vectorizer.get_feature_names_out()
    print("\nList of words:\n", bw_features)
    print(bw_vectorized.toarray())
    print([(bw_features[i], n) for i, n in enumerate(bw_vectorized.toarray()[0]) if n > 0])

    # vectorize text with Tf-Idf algorithm
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorized = tfidf_vectorizer.fit_transform(documents)

    print("\nshape of the vectorized dataframe:\n", tfidf_vectorized.shape)
    tfidf_features = tfidf_vectorizer.get_feature_names_out()
    print("\nList of words:\n", tfidf_features)
    print(tfidf_vectorized.toarray())
    print([(tfidf_features[i], n) for i, n in enumerate(tfidf_vectorized.toarray()[0]) if n > 0])

    return bw_vectorized, tfidf_vectorized


def shuffle_and_split(vectirized, df: pd.DataFrame):
    joined: pd.DataFrame = pd.DataFrame.sparse.from_spmatrix(vectirized)
    joined["CLASS"] = df["CLASS"]
    shuffled = joined.sample(frac=1)

    trainig_amount = round(shuffled.shape[0] * 0.75)
    training = shuffled[:trainig_amount]
    test = shuffled[trainig_amount:]

    training_features = training.drop(
        labels=["CLASS"],
        axis=1,
    )
    training_target = training["CLASS"]

    test_features = test.drop(
        labels=["CLASS"],
        axis=1,
    )
    test_target = test["CLASS"]

    return training_features, training_target, test_features, test_target


def main():
    comments_df = load_data()
    check_basic_data(comments_df)

    tokens_df = pre_precess(comments_df)
    _, tfidf_vectorized = vectorize_tokens(tokens_df)
    training_feature, training_target, test_feature, test_target = shuffle_and_split(tfidf_vectorized, tokens_df)

    model = MultinomialNB()
    model.fit(training_feature, training_target)

    predictions = model.predict(test_feature)

    print("Accuracy: ", accuracy_score(test_target, predictions))
    print("Confusion matrix: ", confusion_matrix(test_target, predictions))


if __name__ == "__main__":
    main()
