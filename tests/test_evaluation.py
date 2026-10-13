import math
from src.evaluation import precision_recall_ndcg


def test_worked_example():
    recs = [1, 99, 98, 2, 97, 96, 95, 94, 93, 92]   # liked movies 1 and 2 at positions 1 and 4
    relevant = {1, 2, 3}
    p, r, n = precision_recall_ndcg(recs, relevant, k=10)
    assert math.isclose(p, 0.2)
    assert math.isclose(r, 2 / 3)
    assert math.isclose(n, 0.6714, abs_tol=1e-3)