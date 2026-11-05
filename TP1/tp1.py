from pathlib import Path
from collections import Counter, defaultdict
from math import log10

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer, LancasterStemmer


# ==========================================
# 1. READ THE DOCUMENT COLLECTION
# ==========================================

folder = Path(__file__).resolve().parent
dataset_folder = folder.parent / "dataset"

documents = {}

for i in range(1, 7):
    filename = dataset_folder / f"D{i}.txt"

    if not filename.exists():
        raise FileNotFoundError(f"File not found: {filename}")

    with open(filename, "r", encoding="utf-8") as file:
        documents[f"D{i}"] = file.read()


# ==========================================
# 2. INITIALIZE NLP TOOLS
# ==========================================

nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))

tokenizer = RegexpTokenizer(r"\w+")

porter = PorterStemmer()
lancaster = LancasterStemmer()


# ==========================================
# 3. PREPROCESSING FUNCTIONS
# ==========================================

def extract_terms(text, method):

    if method == "split":
        return text.split()

    elif method == "regex":
        return tokenizer.tokenize(text)

    else:
        raise ValueError("Unknown tokenization method")


def remove_stopwords(terms):

    result = []

    for term in terms:
        term = term.lower()

        if term not in stop_words:
            result.append(term)

    return result


def normalize_terms(terms, stemmer):

    return [stemmer.stem(term) for term in terms]


# ==========================================
# 4. BUILD INDEX + CALCULATE TF-IDF
# ==========================================

def build_index(documents, method, stemmer, output_name):

    output_folder = folder / "results" / output_name
    output_folder.mkdir(parents=True, exist_ok=True)

    # Store the processed terms of each document
    processed_documents = {}

    # Store term frequencies for each document
    document_frequencies = {}

    # --------------------------------------
    # STEP 1: Preprocess each document
    # --------------------------------------

    for doc_id, text in documents.items():

        terms = extract_terms(text, method)

        terms = remove_stopwords(terms)

        terms = normalize_terms(terms, stemmer)

        processed_documents[doc_id] = terms

        # Count each term in this document
        document_frequencies[doc_id] = Counter(terms)

    # --------------------------------------
    # STEP 2: Calculate max frequency
    # --------------------------------------

    max_frequencies = {}

    for doc_id, frequencies in document_frequencies.items():

        if frequencies:
            max_frequencies[doc_id] = max(frequencies.values())
        else:
            max_frequencies[doc_id] = 0

    # --------------------------------------
    # STEP 3: Calculate document frequency
    # --------------------------------------

    # DF = number of documents containing a term

    document_frequency = Counter()

    for frequencies in document_frequencies.values():

        for term in frequencies:
            document_frequency[term] += 1

    # Total number of documents
    N = len(documents)

    # --------------------------------------
    # STEP 4: Calculate TF-IDF
    # --------------------------------------

    weights = {}

    for doc_id, frequencies in document_frequencies.items():

        weights[doc_id] = {}

        max_freq = max_frequencies[doc_id]

        for term, freq in frequencies.items():

            # Term Frequency
            if max_freq > 0:
                tf = freq / max_freq
            else:
                tf = 0

            # Inverse Document Frequency
            nt = document_frequency[term]

            idf = log10(1 + N / nt)

            # TF-IDF
            weight = tf * idf

            weights[doc_id][term] = weight

    # --------------------------------------
    # STEP 5: Create descriptor files
    # --------------------------------------

    # Format:
    # Document Term Frequency Weight

    for doc_id, frequencies in document_frequencies.items():

        filename = output_folder / f"{doc_id}.txt"

        with open(filename, "w", encoding="utf-8") as file:

            for term in sorted(frequencies):

                freq = frequencies[term]
                weight = weights[doc_id][term]

                file.write(
                    f"{doc_id} {term} {freq} {weight:.4f}\n"
                )

    # --------------------------------------
    # STEP 6: Create inverted index
    # --------------------------------------

    # Format:
    # Term Document Frequency Weight

    inverted_index = defaultdict(list)

    for doc_id, frequencies in document_frequencies.items():

        for term, freq in frequencies.items():

            weight = weights[doc_id][term]

            inverted_index[term].append(
                (doc_id, freq, weight)
            )

    with open(
        output_folder / "inverted_index.txt",
        "w",
        encoding="utf-8"
    ) as file:

        for term in sorted(inverted_index):

            entries = sorted(
                inverted_index[term],
                key=lambda item: int(item[0][1:])
            )

            for doc_id, freq, weight in entries:

                file.write(
                    f"{term} {doc_id} {freq} {weight:.4f}\n"
                )

    # --------------------------------------
    # STEP 7: Save document frequencies
    # --------------------------------------

    with open(
        output_folder / "document_frequencies.txt",
        "w",
        encoding="utf-8"
    ) as file:

        for term in sorted(document_frequency):

            file.write(
                f"{term} {document_frequency[term]}\n"
            )

    # --------------------------------------
    # STEP 8: Display results
    # --------------------------------------

    print("\n================================")
    print("METHOD:", output_name)
    print("================================")

    print("\nDocument frequencies:")
    print(dict(sorted(document_frequency.items())))

    print("\nTF-IDF weights:")

    for doc_id in weights:

        print(f"\n{doc_id}:")

        for term, weight in sorted(weights[doc_id].items()):

            print(
                f"{term}: frequency = "
                f"{document_frequencies[doc_id][term]}, "
                f"weight = {weight:.4f}"
            )

    print("\nFiles saved in:", output_folder)


# ==========================================
# 5. EXECUTE THE FOUR COMBINATIONS
# ==========================================

build_index(
    documents,
    "split",
    porter,
    "split_porter"
)

build_index(
    documents,
    "split",
    lancaster,
    "split_lancaster"
)

build_index(
    documents,
    "regex",
    porter,
    "regex_porter"
)

build_index(
    documents,
    "regex",
    lancaster,
    "regex_lancaster"
)

print("\nExercise 2 completed successfully!")