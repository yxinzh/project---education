import pandas as pd
import os
import spacy
import pickle
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

sample_size = 10000
vocab_size = 1000

def read_data(sample_size):
    """
    Load and sample funded and non-funded essay data.

    This function reads two CSV files (for funded and non-funded projects),
    samples an equal number of rows from each, and returns them as pandas DataFrames.

    Args:
        sample_size (int): Number of rows to sample from each dataset.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - funded_essay: Sampled DataFrame of fully funded essays.
            - nfunded_essay: Sampled DataFrame of not fully funded essays.
    """
 
    path = os.getcwd()
    data_path = os.path.join(path, 'database')
    funded = os.path.join(data_path, 'essay_funded_essay')
    nfunded = os.path.join(data_path, 'essay_nfunded_essay')
    funded_essay = pd.read_csv(funded)
    nfunded_essay = pd.read_csv(nfunded)

    funded_essay = funded_essay['essay'].dropna().sample(n=sample_size, random_state=6)
    nfunded_essay = nfunded_essay['essay'].dropna().sample(n=sample_size, random_state=6)
   
    funded_essay_df = funded_essay.to_frame()
    funded_essay_df['fully_funded'] = 1
    nfunded_essay_df = nfunded_essay.to_frame()
    nfunded_essay_df['fully_funded'] = 0
    all_essays = pd.concat([funded_essay_df, nfunded_essay_df], ignore_index=True)

    return all_essays


def preprocess(texts):
    """
    Preprocesses a list of text documents using spaCy for lemmatization,
    lowercasing, and removal of non-alphabetic characters, stop words, and short words.

    Args:
        texts (list): A list of strings, where each string is a document to be processed.

    Returns:
        list: A list of strings, where each string contains the preprocessed tokens
              of the corresponding input document, joined by spaces.
    """
    n_process = max(os.cpu_count() - 1, 1) #get the multi thread number
    # 这个地方没有禁用 "tagger","morphologizer"
    nlp = spacy.load('en_core_web_sm',disable=["ner","parser"])
    docs = nlp.pipe(texts,batch_size=500,n_process=n_process)
    results = []
    print("Preprocessing texts with spaCy...")
    for doc in tqdm(docs, total=len(texts)):
        tokens = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop and len(token) > 2]
        results.append(' '.join(tokens))
    return  results

def get_tf(text_corpus, name):
    """
    Converts a corpus of text documents into a Term Frequency (TF) matrix using CountVectorizer.

    Args:
        text_corpus (list): A list of strings, where each string is a preprocessed document.
        name (str): A name to identify this TF representation.

    Returns:
        dict: A dictionary containing the following keys:
            'name' (str): The provided name for this TF representation.
            'matrix' (scipy.sparse._csr.csr_matrix): The resulting TF matrix.
            'vectorizer' (sklearn.feature_extraction.text.CountVectorizer): The fitted CountVectorizer object.
    """
    vectorizer = CountVectorizer(max_df=0.90, min_df=10, max_features=vocab_size)
    tf = vectorizer.fit_transform(text_corpus)
    return {'name': name, 'matrix': tf, 'vectorizer': vectorizer}

def get_tfidf(text_corpus, name):
    """
    Converts a corpus of text documents into a Term Frequency-Inverse Document Frequency (TF-IDF) matrix.

    Args:
        text_corpus (list): A list of strings, where each string is a preprocessed document.
        name (str): A name to identify this TF-IDF representation.

    Returns:
        dict: A dictionary containing the following keys:
            'name' (str): The provided name for this TF-IDF representation.
            'matrix' (scipy.sparse._csr.csr_matrix): The resulting TF-IDF matrix.
            'vectorizer' (sklearn.feature_extraction.text.TfidfVectorizer): The fitted TfidfVectorizer object.
    """
    vectorizer = TfidfVectorizer(max_df = 0.90, min_df =10, max_features=vocab_size)
    tfidf = vectorizer.fit_transform(text_corpus)
    return {'name': name, 'matrix': tfidf, 'vectorizer': vectorizer}


if __name__ == "__main__":
    print("Reading data...")
    all_essays = read_data(sample_size)
    texts = all_essays['essay'].tolist()
    
    print("Preprocessing all essays...")
    processed_texts = preprocess(texts)

    print("Generating TF vectors...")
    tf_data = get_tf(processed_texts, 'all_essays_tf')

    print("Generating TF-IDF vectors...")
    tfidf_data = get_tfidf(processed_texts, 'all_essays_tfidf')

    print("Saving processed data...")
    processed_data = {
        tf_data['name']: {
            'matrix': tf_data['matrix'],
            'vectorizer': tf_data['vectorizer']
        },
        tfidf_data['name']: {
            'matrix': tfidf_data['matrix'],
            'vectorizer': tfidf_data['vectorizer']
        },
        'labels': all_essays['fully_funded'].values 
    }

    output_dir = 'essays_processed_data_named'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'full_processed_features.pkl')

    with open(output_path, 'wb') as f:
        pickle.dump(processed_data, f)

    print(f"\nAll processed features saved to '{output_path}'.")

