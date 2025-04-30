import pandas as pd
import os
import spacy
import pickle
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

sample_size = 100000
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

    funded_essay = funded_essay['short_description'].dropna().sample(n=sample_size, random_state=6)
    nfunded_essay = nfunded_essay['short_description'].dropna().sample(n=sample_size, random_state=6)

    return funded_essay, nfunded_essay


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
    nlp = spacy.load('en_core_web_sm',disable=["ner","parser","tagger","morphologizer"])
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
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=vocab_size)
    tf = vectorizer.fit_transform(text_corpus)
    return {'name': name, 'matrix': tf, 'vectorizer': vectorizer}

def get_tfidf(text_corpus, name):
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
    vectorizer = TfidfVectorizer(max_df = 0.95, min_df =2, max_features=vocab_size)
    tfidf = vectorizer.fit_transform(text_corpus)
    return {'name': name, 'matrix': tfidf, 'vectorizer': vectorizer}


if __name__ == "__main__":
    print("Reading data...")
    funded_essay, nfunded_essay = read_data(sample_size)

    print("\nPreprocessing funded essays:")
    processed_funded_essay = preprocess(funded_essay)

    print("\nPreprocessing non-funded essays:")
    processed_nfunded_essay = preprocess(nfunded_essay)
    
    print("\nGenerating TF vectors...")
    funded_tf_data = get_tf(processed_funded_essay, 'funded_essay_tf')
    nfunded_tf_data = get_tf(processed_nfunded_essay, 'nfunded_essay_tf')

      # Generate TF-IDF matrices
    print("\nGenerating TF-IDF vectors...")
    funded_tfidf_data = get_tfidf(processed_funded_essay, 'funded_essay_tfidf')
    nfunded_tfidf_data = get_tfidf(processed_nfunded_essay, 'nfunded_essay_tfidf')
   
    print(f"TF Data for '{funded_tf_data['name']}':")
    print("  Matrix Shape:", funded_tf_data['matrix'].shape)
    print("  Vocabulary Size:", len(funded_tf_data['vectorizer'].vocabulary_))

    print(f"TF Data for '{nfunded_tf_data['name']}':")
    print("  Matrix Shape:", nfunded_tf_data['matrix'].shape)
    print("  Vocabulary Size:", len(nfunded_tf_data['vectorizer'].vocabulary_))

    print(f"TF-IDF Data for '{funded_tfidf_data['name']}':")
    print("  Matrix Shape:", funded_tfidf_data['matrix'].shape)
    print("  Vocabulary Size:", len(funded_tfidf_data['vectorizer'].vocabulary_))

    print(f"TF-IDF Data for '{nfunded_tfidf_data['name']}':")
    print("  Matrix Shape:", nfunded_tfidf_data['matrix'].shape)
    print("  Vocabulary Size:", len(nfunded_tfidf_data['vectorizer'].vocabulary_))

    
    processed_data = {
        funded_tf_data['name']: {
            'matrix': funded_tf_data['matrix'],
            'vectorizer': funded_tf_data['vectorizer']
        },
        nfunded_tf_data['name']: {
            'matrix': nfunded_tf_data['matrix'],
            'vectorizer': nfunded_tf_data['vectorizer']
        },
        funded_tfidf_data['name']: {
            'matrix': funded_tfidf_data['matrix'],
            'vectorizer': funded_tfidf_data['vectorizer']
        },
        nfunded_tfidf_data['name']: {
            'matrix': nfunded_tfidf_data['matrix'],
            'vectorizer': nfunded_tfidf_data['vectorizer']
        }
    }

    output_dir = 'processed_data_named'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'processed_features.pkl')

    with open(output_path, 'wb') as f:
        pickle.dump(processed_data, f)

    print(f"\n All processed features saved to '{output_path}'.")
