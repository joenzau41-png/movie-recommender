# MovieLens Recommender System

A movie recommendation system built on the MovieLens dataset, following the CRISP-DM process from business understanding through deployment.

## Business problem

- **Problem:** A streaming platform wants users to spend more time watching by showing each of them a personalised "Recommended for you" row of 10 movies on the home page. Users who find something they like quickly are less likely to leave, so better suggestions directly support engagement and retention.
- **ML task:** Given a user and their past ratings, the model returns a ranked list of 10 movies they have not yet seen. This is a ranking problem rather than a rating-prediction problem, because what the user sees is an ordered list, and only the top few positions matter.
- **Success metrics:** Precision@10, Recall@10 and NDCG@10, measured on each user's most recent ratings (held out by time) and averaged per user. A movie counts as relevant if the user rated it 4.0 or higher. The models must clearly beat two baselines: random recommendations and "most popular". Secondary goal: catalogue coverage, so the system doesn't only recommend blockbusters.
- **Constraints:** Recommendations should be served in under 200 ms, new users with no history need a sensible fallback (popular titles), and rarely rated movies carry too little data to learn from, so they are excluded from training.

## Data

[MovieLens `ml-latest-small`](https://grouplens.org/datasets/movielens/latest/) (GroupLens): 100,836 ratings from 610 users on 9,724 movies. Download it with `python src/download_data.py`.

## Results

Final evaluation on the held-out test set (each user's most recent 15% of ratings), k = 10, 574 users. Models were tuned on the validation set, then refit on train + validation before testing. A movie counts as relevant if rated 4.0 or higher.

| Model | Precision@10 | Recall@10 | NDCG@10 | Coverage |
|---|---|---|---|---|
| Random | 0.0044 | 0.0023 | 0.0050 | 79.6% |
| Most popular | 0.0402 | 0.0472 | 0.0592 | 2.8% |
| Item-kNN (cosine) | 0.0524 | 0.0716 | 0.0730 | 11.8% |
| **SVD (15 components)** | **0.0692** | **0.0913** | **0.0934** | 13.1% |

SVD improves on the most-popular baseline by about 72% in precision@10 and 58% in NDCG@10, and on item-kNN by about 32% and 28%. It leads for both light and heavy users (split at the median number of ratings). A hybrid with genre-based content features did not improve on pure SVD.

## Limitations

- TruncatedSVD treats unrated movies as zeros rather than missing; a method fitted only on observed ratings (for example ALS) may do better.
- Movies with fewer than 5 ratings are excluded, so brand-new titles cannot be recommended.
- The per-user time split means one user's test period can overlap another's training period.
- Evaluation is offline only; real impact would need an A/B test.

## Project structure

- `notebooks/`: EDA, data preparation, baselines and models
- `src/`: reusable code (data download, evaluation metrics)
- `tests/`: unit tests

## Progress

- [x] Business understanding
- [x] Data understanding (EDA)
- [x] Data preparation (per-user chronological split)
- [x] Baselines and evaluation framework
- [x] Collaborative filtering and matrix factorization models
- [x] Final evaluation and model comparison
- [ ] FastAPI deployment and Docker

## How to run

    python -m venv .venv
    source .venv/bin/activate    # Windows (Git Bash): source .venv/Scripts/activate
    pip install -r requirements.txt
    python src/download_data.py

Then run the notebooks in order.