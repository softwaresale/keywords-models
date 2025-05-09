import json
import re


def normalize_keyword(kw):
    kw = kw.lower()
    kw = re.sub(r"[^a-z0-9\s]", "", kw)
    kw = re.sub(r"\s+", " ", kw)
    return kw.strip()


def is_fuzzy_match(pred, true_keywords):
    for true_kw in true_keywords:
        if pred in true_kw or true_kw in pred:
            return True
    return False


def average_precision(predicted_keywords, true_keywords):
    true_keywords = [normalize_keyword(kw) for kw in true_keywords]
    predicted_keywords = [normalize_keyword(kw) for kw in predicted_keywords]

    correct = 0
    precisions = []

    for i, pred_kw in enumerate(predicted_keywords):
        if is_fuzzy_match(pred_kw, true_keywords):
            correct += 1
            precision_at_i = correct / (i + 1)
            precisions.append(precision_at_i)

    if not precisions:
        return 0.0
    return sum(precisions) / len(true_keywords)


def mean_average_precision(data, field="top_keywords_combined"):
    total_ap = 0.0

    for obj in data:
        true_kw = obj["keywords"]
        pred_kw = obj[field]
        ap = average_precision(pred_kw, true_kw)
        total_ap += ap

    return total_ap / len(data)


def main():
    INPUT_PATH = "data/training-data-5k-tfidf-2keywords.json"

    print("Loading data...")
    with open(INPUT_PATH, "r") as f:
        data = json.load(f)

    map_combined = mean_average_precision(data, field="top_keywords_combined")
    map_abstract = mean_average_precision(data, field="top_keywords_abstract")
    map_content = mean_average_precision(data, field="top_keywords_content")

    print(f"MAP (abstract): {map_abstract:.4f}")
    print(f"MAP (content): {map_content:.4f}")


if __name__ == "__main__":
    main()
