import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer

PATH_TO_DATA = "data/training-data.ndjson"
OUTPUT_PATH = "data/training-data-tfidf-keywords.json"


def clean_text(text):

    text = text.encode("ascii", "ignore").decode("ascii")

    text = re.sub(r"(\w)-\s*(\w)", r"\1\2", text)

    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def read_file():
    max_val = float("-inf")
    sum_val = 0
    j = 0
    data = []

    with open(PATH_TO_DATA, "r") as f:
        for line in f:
            obj = json.loads(line)
            data.append(obj)
            max_val = max(max_val, len(obj["keywords"]))
            sum_val += len(obj["keywords"])
            j += 1

    avg_keywords = round(sum_val / j)
    print(f"Max keywords in a document: {max_val}")
    print(f"Total keywords: {sum_val}")
    print(f"Average keywords per document (rounded): {avg_keywords}")

    return data, avg_keywords


def build_document_lists(data):
    abstracts = []
    contents = []
    combined = []

    for obj in data:
        abstracts.append(obj["abstract_content"])
        contents.append(obj["content"])
        combined.append(obj["content"] + " " + obj["abstract_content"])

    return abstracts, contents, combined


def clean_original_keywords(keywords):
    cleaned = []
    for kw in keywords:
        if ";" in kw:
            parts = kw.split(";")
            parts = [p.strip() for p in parts if p.strip()]  # clean and remove empty
            cleaned.extend(parts)
        else:
            cleaned.append(kw.strip())
    return cleaned


def run_tfidf(documents, top_k):
    # First clean the documents
    cleaned_documents = [clean_text(doc) for doc in documents]

    # TF-IDF with unigrams, igrams, trigrams
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000,
        ngram_range=(1, 3),  # allow 1-word, 2-word, 3-word phrases
    )
    tfidf_matrix = vectorizer.fit_transform(cleaned_documents)
    feature_names = vectorizer.get_feature_names_out()

    top_keywords = []
    for i in range(tfidf_matrix.shape[0]):
        tfidf_scores = tfidf_matrix[i].toarray().flatten()
        top_indices = tfidf_scores.argsort()[::-1][:top_k]
        keywords = [feature_names[idx] for idx in top_indices]
        top_keywords.append(keywords)

    return top_keywords


def main():
    print("Executing Control Baseline TF-IDF Keyword Identifier")
    data, avg_keywords = read_file()
    abstracts, contents, combined = build_document_lists(data)

    abstract_keywords = run_tfidf(abstracts, avg_keywords)

    content_keywords = run_tfidf(contents, avg_keywords)

    combined_keywords = run_tfidf(combined, avg_keywords)

    final_output = []
    for i, obj in enumerate(data):
        new_obj = {
            "arxiv_id": obj["arxiv_id"],
            "keywords": clean_original_keywords(obj["keywords"]),
            "top_keywords_abstract": abstract_keywords[i],
            "top_keywords_content": content_keywords[i],
            "top_keywords_combined": combined_keywords[i],
        }
        final_output.append(new_obj)

    print(f"Saving to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(final_output, f, indent=2)


if __name__ == "__main__":
    main()
