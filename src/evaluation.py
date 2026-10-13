import numpy as np


def precision_recall_ndcg(recommended, relevant, k):
    recommended = recommended[:k]
    hits = [1 if item in relevant else 0 for item in recommended]
    n_hits = sum(hits)

    precision = n_hits / k
    recall = n_hits / len(relevant)

    dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
    ideal_hits = min(len(relevant), k)
    idcg = sum(1 / np.log2(i + 2) for i in range(ideal_hits))
    ndcg = dcg / idcg
    return precision, recall, ndcg

def evaluate(recommend_fn, history, target, k=10, threshold=4.0):
    seen_by_user = history.groupby("userId")["movieId"].apply(set).to_dict()
    relevant_by_user = (
        target[target["rating"] >= threshold]
        .groupby("userId")["movieId"].apply(set).to_dict()
    )

    scores, all_recommended = [], set()
    for user_id, relevant in relevant_by_user.items():
        seen = seen_by_user.get(user_id, set())
        recs = recommend_fn(user_id, seen, k)
        all_recommended.update(recs)
        scores.append(precision_recall_ndcg(recs, relevant, k))

    scores = np.array(scores)
    return {
        "precision@k": scores[:, 0].mean(),
        "recall@k": scores[:, 1].mean(),
        "ndcg@k": scores[:, 2].mean(),
        "users_evaluated": len(scores),
        "n_unique_items_recommended": len(all_recommended),
    }