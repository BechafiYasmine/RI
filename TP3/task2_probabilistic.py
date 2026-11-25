

from pathlib import Path
from math import log
import re

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

K1 = 1.5
B = 0.75

TEST_QUERIES = {
    "q1": "large language models for information retrieval and ranking",
    "q2": "LLM for information retrieval and Ranking",
    "q3": "query Reformulation in information retrieval",
    "q4": "ranking Documents",
    "q5": "Optimizing recommendation systems with LLMs by leveraging item metadata",
}

# Relevant documents supplied in the LAB 3 handout (page 3).
RELEVANT_DOCUMENTS = {
    "q1": ["D2", "D4"],
    "q2": ["D2", "D4"],
    "q3": ["D4", "D1"],
    "q4": ["D2", "D1"],
    "q5": ["D3", "D6"],
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


def load_document_data():
    """
    Load TF-IDF weights and term frequencies.

    Each line must contain:
        document_id term frequency weight
    """
    directory = find_index_directory()

    document_vectors = {doc_id: {} for doc_id in DOCUMENT_IDS}
    term_frequencies = {doc_id: {} for doc_id in DOCUMENT_IDS}

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

                term_frequencies[doc_id][term] = int(frequency)
                document_vectors[doc_id][term] = float(weight)

    print(f"Loaded LAB 1 files from: {directory}")
    return document_vectors, term_frequencies


# ============================================================
# 4. BUILD VOCABULARY AND BINARY TERM-DOCUMENT INFORMATION
# ============================================================

def build_collection_statistics(document_vectors, term_frequencies):
    """Build document frequency, document lengths, and collection statistics."""
    vocabulary = sorted({
        term
        for vector in document_vectors.values()
        for term in vector
    })

    number_of_documents = len(DOCUMENT_IDS)

    document_frequency = {}
    for term in vocabulary:
        document_frequency[term] = sum(
            1 for doc_id in DOCUMENT_IDS
            if term in term_frequencies[doc_id]
        )

    document_lengths = {
        doc_id: sum(term_frequencies[doc_id].values())
        for doc_id in DOCUMENT_IDS
    }

    average_document_length = (
        sum(document_lengths.values()) / number_of_documents
        if number_of_documents else 0.0
    )

    return (
        vocabulary,
        document_frequency,
        document_lengths,
        average_document_length,
    )


# ============================================================
# 5. QUERY PREPROCESSING
# ============================================================

def build_query_terms(query):
    """Return unique preprocessed query terms (binary query representation)."""
    return sorted(set(preprocess(query)))


# ============================================================
# 6. CLASSIC BIR - PROBABILISTIC TERM WEIGHTS
# ============================================================

def classic_bir_term_weight(term, document_frequency, relevant_docs=None):
    """
    Compute the smoothed Binary Independence Model log-odds weight.

    Without learning:
        p = 0.5
        u = (n_i + 0.5) / (N + 1)

    With learning:
        p = (r_i + 0.5) / (R + 1)
        u = (n_i - r_i + 0.5) / (N - R + 1)
    """
    N = len(DOCUMENT_IDS)
    n_i = document_frequency.get(term, 0)

    if relevant_docs is None:
        p_i = 0.5
        u_i = (n_i + 0.5) / (N + 1.0)
    else:
        relevant_set = set(relevant_docs)
        R = len(relevant_set)

        r_i = sum(
            1 for doc_id in relevant_set
            if term in CURRENT_TERM_FREQUENCIES[doc_id]
        )

        p_i = (r_i + 0.5) / (R + 1.0)
        u_i = (n_i - r_i + 0.5) / (N - R + 1.0)

    # Smoothing keeps p_i and u_i strictly between 0 and 1.
    epsilon = 1e-12
    p_i = min(max(p_i, epsilon), 1.0 - epsilon)
    u_i = min(max(u_i, epsilon), 1.0 - epsilon)

    return log((p_i * (1.0 - u_i)) / (u_i * (1.0 - p_i)))


def classic_bir_score(query_terms, document_id, document_frequency,
                      relevant_docs=None):
    """
    Classic BIR score for one document.

    A query term contributes its probabilistic weight if it occurs
    in the document; otherwise it contributes zero.
    """
    score = 0.0

    for term in query_terms:
        if term in CURRENT_TERM_FREQUENCIES[document_id]:
            score += classic_bir_term_weight(
                term, document_frequency, relevant_docs
            )

    return score


def rank_classic_bir(query_terms, document_frequency, relevant_docs=None):
    results = []

    for doc_id in DOCUMENT_IDS:
        score = classic_bir_score(
            query_terms, doc_id, document_frequency, relevant_docs
        )
        results.append((doc_id, score))

    return sorted(results, key=lambda item: (-item[1], item[0]))


# ============================================================
# 7. EXTENDED BIR - TF-IDF WEIGHTED DOCUMENT TERMS
# ============================================================

def extended_bir_score(query_terms, document_id, document_vectors,
                       document_frequency, relevant_docs=None):
    """
    Extended BIR score.

    The classic probabilistic term weight is multiplied by the
    document's TF-IDF weight for that query term.

    This uses TF-IDF values as the document-term weights, as requested
    in the LAB 3 handout.
    """
    score = 0.0

    for term in query_terms:
        tfidf_weight = document_vectors[document_id].get(term, 0.0)

        if tfidf_weight != 0.0:
            probabilistic_weight = classic_bir_term_weight(
                term, document_frequency, relevant_docs
            )
            score += tfidf_weight * probabilistic_weight

    return score


def rank_extended_bir(query_terms, document_vectors,
                      document_frequency, relevant_docs=None):
    results = []

    for doc_id in DOCUMENT_IDS:
        score = extended_bir_score(
            query_terms,
            doc_id,
            document_vectors,
            document_frequency,
            relevant_docs,
        )
        results.append((doc_id, score))

    return sorted(results, key=lambda item: (-item[1], item[0]))


# ============================================================
# 8. BM25
# ============================================================

def bm25_score(query_terms, document_id, term_frequencies,
               document_frequency, document_lengths,
               average_document_length):
    """
    BM25 score:

        IDF(t) = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

        BM25(D,Q) = sum over query terms of:
          IDF(t) * [ tf(t,D) * (K1 + 1) ] /
          [ tf(t,D) + K1 * (1 - B + B * |D| / avgdl) ]

    K1 = 1.5 and B = 0.75, as specified in the handout.
    """
    N = len(DOCUMENT_IDS)
    doc_length = document_lengths[document_id]

    score = 0.0

    for term in query_terms:
        tf = term_frequencies[document_id].get(term, 0)
        df = document_frequency.get(term, 0)

        if tf == 0:
            continue

        idf = log(
            1.0 + (N - df + 0.5) / (df + 0.5)
        )

        if average_document_length == 0:
            length_normalization = 1.0
        else:
            length_normalization = (
                K1 * (
                    1.0 - B
                    + B * doc_length / average_document_length
                )
            )

        term_score = (
            idf * (tf * (K1 + 1.0)) / (tf + length_normalization)
        )

        score += term_score

    return score


def rank_bm25(query_terms, term_frequencies, document_frequency,
              document_lengths, average_document_length):
    results = []

    for doc_id in DOCUMENT_IDS:
        score = bm25_score(
            query_terms,
            doc_id,
            term_frequencies,
            document_frequency,
            document_lengths,
            average_document_length,
        )
        results.append((doc_id, score))

    return sorted(results, key=lambda item: (-item[1], item[0]))


# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

def print_ranked_results(results, score_name="RSV"):
    print(f"{'Rank':<8}{'Document':<15}{score_name}")
    print("-" * 38)

    for rank, (doc_id, score) in enumerate(results, start=1):
        print(f"{rank:<8}{doc_id:<15}{score:.6f}")


def print_interpretation(results, relevant_docs):
    """Briefly compare the top-ranked documents with the supplied relevance set."""
    relevant_set = set(relevant_docs)
    top_document = results[0][0] if results else None

    if top_document in relevant_set:
        print(
            f"Interpretation: The top-ranked document ({top_document}) "
            "is among the documents marked relevant in the handout."
        )
    else:
        print(
            f"Interpretation: The top-ranked document ({top_document}) "
            "is not in the relevance set supplied in the handout."
        )


# ============================================================
# 10. EXECUTE TASK 2
# ============================================================

def execute_task_2(document_vectors, term_frequencies,
                   document_frequency, document_lengths,
                   average_document_length):
    for query_id, query_text in TEST_QUERIES.items():
        query_terms = build_query_terms(query_text)

        print("\n" + "=" * 70)
        print(f"{query_id.upper()}: {query_text}")
        print("Preprocessed binary query terms:", query_terms)

        # 1. Classic BIR without learning data
        print("\n--- 1. Classic BIR - Without Learning Data ---")
        results = rank_classic_bir(
            query_terms, document_frequency, relevant_docs=None
        )
        print_ranked_results(results, "RSV")

        # 2. Classic BIR with learning data
        relevant_docs = RELEVANT_DOCUMENTS[query_id]
        print("\n--- 2. Classic BIR - With Learning Data ---")
        print("Relevant documents used for learning:", relevant_docs)
        results = rank_classic_bir(
            query_terms, document_frequency, relevant_docs=relevant_docs
        )
        print_ranked_results(results, "RSV")
        print_interpretation(results, relevant_docs)

        # 3. Extended BIR without learning data
        print("\n--- 3. Extended BIR - Without Learning Data ---")
        results = rank_extended_bir(
            query_terms,
            document_vectors,
            document_frequency,
            relevant_docs=None,
        )
        print_ranked_results(results, "RSV")

        # 4. Extended BIR with learning data
        print("\n--- 4. Extended BIR - With Learning Data ---")
        print("Relevant documents used for learning:", relevant_docs)
        results = rank_extended_bir(
            query_terms,
            document_vectors,
            document_frequency,
            relevant_docs=relevant_docs,
        )
        print_ranked_results(results, "RSV")
        print_interpretation(results, relevant_docs)

        # 5. BM25
        print("\n--- 5. BM25 (K1=1.5, b=0.75) ---")
        results = rank_bm25(
            query_terms,
            term_frequencies,
            document_frequency,
            document_lengths,
            average_document_length,
        )
        print_ranked_results(results, "BM25")


# ============================================================
# 11. MAIN
# ============================================================

# Used by the BIR functions to count term occurrences in each document.
CURRENT_TERM_FREQUENCIES = {}


def main():
    global CURRENT_TERM_FREQUENCIES

    document_vectors, term_frequencies = load_document_data()
    CURRENT_TERM_FREQUENCIES = term_frequencies

    (
        vocabulary,
        document_frequency,
        document_lengths,
        average_document_length,
    ) = build_collection_statistics(document_vectors, term_frequencies)

    print("\nDocuments loaded:", ", ".join(DOCUMENT_IDS))
    print("Vocabulary size:", len(vocabulary))
    print("Document lengths:", document_lengths)
    print(f"Average document length: {average_document_length:.2f}")

    execute_task_2(
        document_vectors,
        term_frequencies,
        document_frequency,
        document_lengths,
        average_document_length,
    )

    print("\n" + "=" * 70)
    print("LAB 3 - TASK 2 (PROBABILISTIC MODELS) COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()


"""
LAB 3 - TASK 2: PROBABILISTIC INFORMATION RETRIEVAL MODELS
USTHB - M2 SII - Information Retrieval - 2026/2027

Implemented models:
    1. Classic BIR without learning data
    2. Classic BIR with learning data
    3. Extended BIR without learning data
    4. Extended BIR with learning data
    5. BM25

Uses the Document-Term files generated in LAB 1:
    <document_id> <term> <frequency> <TF-IDF_weight>

Default expected location:
    <project_folder>/TP1/results/regex_porter/D1.txt ... D6.txt

Run:
    python task2_probabilistic.py

IMPORTANT FORMULA NOTE:
The LAB 3 handout refers to smoothing/probabilistic equations from the
lecture notes but does not print those equations. This implementation uses
the standard smoothed Binary Independence Model (BIM) log-odds formula:

    w_i = log( p_i * (1 - u_i) / (u_i * (1 - p_i)) )

Without learning:
    p_i = 0.5
    u_i = (n_i + 0.5) / (N + 1)

With learning:
    p_i = (r_i + 0.5) / (R + 1)
    u_i = (n_i - r_i + 0.5) / (N - R + 1)

For Extended BIR, each query-term log-odds contribution is multiplied
by that document's LAB 1 TF-IDF weight for the term. This is an explicit
implementation choice based on the handout's instruction to use TF-IDF
document-term weights. If your lecture notes give a different Extended
BIR equation, replace extended_bir_score() accordingly.
"""