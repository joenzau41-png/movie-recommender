import pandas as pd
from src.models import (RatingsData, make_popular_recommender,
                        make_score_recommender, svd_scores)


def _toy():
    return pd.DataFrame({
        "userId":  [1, 1, 1, 2, 2, 3, 3, 3],
        "movieId": [10, 20, 30, 10, 20, 20, 30, 40],
        "rating":  [5, 4, 3, 5, 4, 2, 5, 4],
    })


def test_never_recommends_seen_movies():
    data = RatingsData(_toy())
    rec = make_score_recommender(data, svd_scores(data, n_components=2))
    seen = {10, 20}
    assert not set(rec(1, seen, 2)) & seen


def test_returns_k_items():
    data = RatingsData(_toy())
    rec = make_score_recommender(data, svd_scores(data, n_components=2))
    assert len(rec(2, {10, 20}, 2)) == 2


def test_unknown_user_gets_popular_titles():
    data = RatingsData(_toy())
    rec = make_score_recommender(data, svd_scores(data, n_components=2))
    assert rec(999, set(), 2) == make_popular_recommender(data)(999, set(), 2)