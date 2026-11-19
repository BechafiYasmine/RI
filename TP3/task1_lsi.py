"""
LAB 3 - TASK 1: LATENT SEMANTIC INDEXING (LSI)
USTHB - M2 SII - Information Retrieval - 2026/2027

Uses the Document-Term files generated in LAB 1:
    <document_id> <term> <frequency> <TF-IDF_weight>

Default expected location:
    <project_folder>/TP1/results/regex_porter/D1.txt ... D6.txt

Run:
    python task1_lsi.py

If your LAB 1 files are stored elsewhere, change INDEX_DIR below.
"""

from pathlib import Path
from math import sqrt

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Change this path if your LAB 1 results are stored elsewhere.
INDEX_DIR = BASE_DIR.parent / "TP1" / "results" / "regex_porter"

DOCUMENT_IDS = [f"D{i}" for i in range(1, 7)]
K = 3

TEST_QUERIES = {
    "q1": "large language models for information retrieval and ranking",
    "q2": "LLM for information retrieval and Ranking",
    "q3": "query Reformulation in information retrieval",
    "q4": "ranking Documents",
    "q5": "Optimizing recommendation systems with LLMs by leveraging item metadata",
}


# ============================================================
# 2. NLP TOOLS - SAME PIPELINE AS LAB 1 / LAB 2
# ============================================================

nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
TOKENIZER = RegexpTokenizer(r"\w+")
STEMMER = PorterStemmer()


def preprocess(text):
    """Tokenize, lowercase, remove English stop words, and apply Porter stemming."""
    tokens = TOKENIZER.tokenize(text)
    tokens = [token.lower() for token in tokens]
    tokens = [token for token in tokens if token not in STOP_WORDS]
    return [STEMMER.stem(token) for token in tokens]


# ============================================================
# 3. FIND AND LOAD LAB 1 DOCUMENT-TERM FILES
# ============================================================

def find_index_directory():
    """Find the folder containing D1.txt ... D6.txt."""
    possible_dirs = [
        INDEX_DIR,
        BASE_DIR.parent / "TP1" / "results" / "split_porter",
        BASE_DIR / "results" / "regex_porter",
        BASE_DIR / "regex_porter",
        BASE_DIR / "results" / "split_porter",
    ]

    for directory in possible_dirs:
        if directory.exists() and all(
            (directory / f"{doc_id}.txt").exists()
            for doc_id in DOCUMENT_IDS
        ):
            return directory

    raise FileNotFoundError(
        "Could not find the six LAB 1 Document-Term files.\n"
        f"Expected them in: {INDEX_DIR}\n"
        "Change INDEX_DIR at the beginning of this script."
    )


def load_document_vectors():
    """
    Load TF-IDF vectors from the six Document-Term files.

    Each line must contain:
        document_id term frequency weight
    """
    directory = find_index_directory()
    document_vectors = {doc_id: {} for doc_id in DOCUMENT_IDS}

    for doc_id in DOCUMENT_IDS:
        file_path = directory / f"{doc_id}.txt"

        with open(file_path, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                parts = line.strip().split()

                if not parts:
                    continue

                if len(parts) != 4:
                    raise ValueError(
                        f"Invalid line {line_number} in {file_path}:\n{line}"
                    )

                file_doc_id, term, frequency, weight = parts

                if file_doc_id != doc_id:
                    raise ValueError(
                        f"Expected {doc_id}, found {file_doc_id} "
                        f"in {file_path}, line {line_number}."
                    )

                document_vectors[doc_id][term] = float(weight)

    print(f"Loaded LAB 1 files from: {directory}")
    return document_vectors


# ============================================================
# 4. BUILD THE TF-IDF MATRIX W
# ============================================================

def build_tfidf_matrix(document_vectors):
    """
    Build W with:
        rows    = unique indexed terms
        columns = documents D1 ... D6
        values  = TF-IDF weights
    """
    vocabulary = sorted({
        term
        for vector in document_vectors.values()
        for term in vector
    })

    term_to_index = {
        term: index for index, term in enumerate(vocabulary)
    }

    W = np.zeros((len(vocabulary), len(DOCUMENT_IDS)), dtype=float)

    for document_index, doc_id in enumerate(DOCUMENT_IDS):
        for term, weight in document_vectors[doc_id].items():
            term_index = term_to_index[term]
            W[term_index, document_index] = weight

    return W, vocabulary, term_to_index


# ============================================================
# 5. SINGULAR VALUE DECOMPOSITION AND DIMENSION REDUCTION
# ============================================================

def apply_svd(W):
    """
    Decompose W:
        W = U * S * VT

    Keep only the first K singular values and vectors.
    """
    U, S, VT = np.linalg.svd(W, full_matrices=False)

    k = min(K, len(S))

    U_k = U[:, :k]
    S_k = S[:k]
    VT_k = VT[:k, :]

    return U_k, S_k, VT_k


# ============================================================
# 6. BUILD AND PROJECT QUERY VECTORS
# ============================================================

def build_query_vector(query, vocabulary, term_to_index):
    """
    LAB 3 specifies binary query weights:
        1 if a preprocessed term occurs in the query, otherwise 0.
    """
    query_terms = set(preprocess(query))
    query_vector = np.zeros(len(vocabulary), dtype=float)

    for term in query_terms:
        if term in term_to_index:
            query_vector[term_to_index[term]] = 1.0

    return query_vector, sorted(query_terms)


def project_documents(S_k, VT_k):
    """
    Coordinates of documents in the reduced latent space:
        D = S_k * VT_k
    Each column represents one document.
    """
    return np.diag(S_k) @ VT_k


def project_query(query_vector, U_k, S_k):
    """
    Project a query into the reduced latent space:
        q_latent = q^T * U_k * S_k^(-1)

    A zero singular value is safely replaced by zero in the inverse.
    """
    inverse_values = np.zeros_like(S_k)
    nonzero = S_k > 1e-12
    inverse_values[nonzero] = 1.0 / S_k[nonzero]

    return (query_vector @ U_k) * inverse_values


# ============================================================
# 7. COSINE SIMILARITY IN THE LATENT SPACE
# ============================================================

def cosine_similarity(vector_a, vector_b):
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(vector_a, vector_b) / (norm_a * norm_b))


def rank_documents(query_latent, document_latent):
    """Rank all documents by decreasing cosine similarity."""
    results = []

    for document_index, doc_id in enumerate(DOCUMENT_IDS):
        document_vector = document_latent[:, document_index]
        score = cosine_similarity(query_latent, document_vector)
        results.append((doc_id, score))

    return sorted(results, key=lambda item: (-item[1], item[0]))


# ============================================================
# 8. DISPLAY RESULTS
# ============================================================

def print_ranked_results(results):
    print(f"{'Rank':<8}{'Document':<15}{'Similarity'}")
    print("-" * 38)

    for rank, (doc_id, score) in enumerate(results, start=1):
        print(f"{rank:<8}{doc_id:<15}{score:.6f}")


# ============================================================
# 9. EXECUTE LSI FOR ALL TEST QUERIES
# ============================================================

def execute_lsi(document_vectors):
    W, vocabulary, term_to_index = build_tfidf_matrix(document_vectors)

    print("\nTF-IDF MATRIX")
    print("Matrix shape (terms x documents):", W.shape)
    print("Number of indexed terms:", len(vocabulary))

    U_k, S_k, VT_k = apply_svd(W)

    print("\nSVD")
    print("Reduced dimension K:", len(S_k))
    print("U_k shape:", U_k.shape)
    print("S_k shape:", S_k.shape)
    print("VT_k shape:", VT_k.shape)

    document_latent = project_documents(S_k, VT_k)

    print("\nDOCUMENT COORDINATES IN LATENT SPACE")
    print("Shape:", document_latent.shape)

    for query_id, query_text in TEST_QUERIES.items():
        query_vector, query_terms = build_query_vector(
            query_text, vocabulary, term_to_index
        )

        query_latent = project_query(query_vector, U_k, S_k)
        results = rank_documents(query_latent, document_latent)

        print("\n" + "=" * 65)
        print(f"{query_id.upper()}: {query_text}")
        print("Preprocessed query terms:", query_terms)
        print("Query vector shape:", query_vector.shape)
        print("Projected query:", np.round(query_latent, 6))

        print("\n--- LSI Ranking (Cosine Similarity) ---")
        print_ranked_results(results)


# ============================================================
# 10. MAIN
# ============================================================

def main():
    document_vectors = load_document_vectors()
    print("Documents loaded:", ", ".join(document_vectors.keys()))
    execute_lsi(document_vectors)

    print("\n" + "=" * 65)
    print("LAB 3 - TASK 1 (LSI) COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
