import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


class RatingsData:
    """The user-item matrix plus lookups between IDs and row/column numbers."""

    def __init__(self, ratings):
        self.user_ids = np.sort(ratings["userId"].unique())
        self.item_ids = np.sort(ratings["movieId"].unique())
        self.user_index = {u: i for i, u in enumerate(self.user_ids)}
        self.item_index = {m: i for i, m in enumerate(self.item_ids)}
        rows = ratings["userId"].map(self.user_index)
        cols = ratings["movieId"].map(self.item_index)
        self.R = csr_matrix(
            (ratings["rating"], (rows, cols)),
            shape=(len(self.user_ids), len(self.item_ids)),
        )
        self.popularity_rank = (
            ratings.groupby("movieId").size().sort_values(ascending=False).index.tolist()
        )


def make_random_recommender(data, seed=42):
    rng = np.random.default_rng(seed)

    def recommend(user_id, seen, k):
        candidates = [m for m in data.item_ids if m not in seen]
        return list(rng.choice(candidates, size=k, replace=False))
    return recommend


def make_popular_recommender(data):
    def recommend(user_id, seen, k):
        recs = []
        for movie_id in data.popularity_rank:
            if movie_id not in seen:
                recs.append(movie_id)
                if len(recs) == k:
                    break
        return recs
    return recommend


def make_score_recommender(data, scores):
    fallback = make_popular_recommender(data)

    def recommend(user_id, seen, k):
        u = data.user_index.get(user_id)
        if u is None:                       # new user: fall back to popularity
            return fallback(user_id, seen, k)
        user_scores = scores[u].copy()
        seen_idx = [data.item_index[m] for m in seen if m in data.item_index]
        user_scores[seen_idx] = -np.inf     # never recommend seen movies
        top = np.argpartition(-user_scores, k)[:k]
        top = top[np.argsort(-user_scores[top])]
        return data.item_ids[top].tolist()
    return recommend


def itemknn_scores(data):
    S = cosine_similarity(data.R.T)
    return data.R @ S


def svd_scores(data, n_components=15, seed=42):
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    return svd.fit_transform(data.R) @ svd.components_