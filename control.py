import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer

PATH_TO_DATA = "data/training-data.ndjson"
OUTPUT_PATH = "data/training-data-5k-tfidf-2keywords.json"

def clean_text(text):
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"(\w)-\s*(\w)", r"\1\2", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def clean_original_keywords(keywords):
    cleaned = []
    for kw in keywords:
        if ";" in kw:
            parts = kw.split(";")
            parts = [p.strip() for p in parts if p.strip()]
            cleaned.extend(parts)
        else:
            cleaned.append(kw.strip())
    return cleaned

def read_file():
    data = []
    sum_val = 0
    with open(PATH_TO_DATA, "r") as f:
        for line in f:
            obj = json.loads(line)
            data.append(obj)
            sum_val += len(obj["keywords"])
    avg_keywords = round(sum_val / len(data))
    print(f"Loaded {len(data)} documents.")
    print(f"Average keywords per document (rounded): {avg_keywords}")
    return data, avg_keywords

def run_tfidf(documents, top_k):
    cleaned_documents = [clean_text(doc) for doc in documents]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2)
    )
    tfidf_matrix = vectorizer.fit_transform(cleaned_documents)
    feature_names = vectorizer.get_feature_names_out()

    top_keywords = []
    for i in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix.getrow(i)
        if row.nnz == 0:
            top_keywords.append([])
            continue
        sorted_indices = row.indices[row.data.argsort()[::-1][:top_k]]
        keywords = [feature_names[idx] for idx in sorted_indices]
        top_keywords.append(keywords)

    return top_keywords

def main():
    print("Executing Control Baseline TF-IDF Keyword Identifier")
    data, avg_keywords = read_file()


    abstracts = [obj["abstract_content"] for obj in data]
    abstract_keywords = run_tfidf(abstracts, avg_keywords)
    del abstracts  


    contents = [obj["content"] for obj in data]
    content_keywords = run_tfidf(contents, avg_keywords)
    del contents  


    combined = [obj["content"] + " " + obj["abstract_content"] for obj in data]
    combined_keywords = run_tfidf(combined, avg_keywords)
    del combined  

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
