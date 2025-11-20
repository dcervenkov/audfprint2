import numpy as np

from audfprint2.core import hash_table, matcher


def _sample_hits():
    # Columns: id, skew_time, hash, orig_time
    return np.array(
        [
            [0, 0, 10, 0],
            [0, 1, 11, 1],
            [1, 0, 10, 0],
            [1, 2, 12, 2],
        ],
        dtype=np.int32,
    )


def test_exact_match_counts_with_time_ranges():
    m = matcher.Matcher()
    m.threshcount = 1
    m.find_time_range = True
    hits = _sample_hits()
    ids = np.array([0, 1], dtype=np.int32)
    rawcounts = np.array([2, 2], dtype=np.int32)

    results = m._exact_match_counts(hits, ids, rawcounts)
    assert results.shape[1] == 7
    # Should contain entries for both ids
    assert set(results[:, 0]) == {0, 1}
    assert np.all(results[:, 1] >= m.threshcount)


def test_match_hashes_returns_sorted_results():
    ht = hash_table.HashTable(hashbits=4, depth=3, maxtime=32)
    ht.store("track", [(0, 10), (1, 11), (2, 12)])

    m = matcher.Matcher()
    m.threshcount = 1
    # Force approximate counting path
    m.exact_count = False

    query_hashes = np.array([[0, 10], [1, 11], [2, 12]], dtype=np.int32)
    results = m.match_hashes(ht, query_hashes)

    assert results.shape[1] == 7
    # Sorted by filtered count descending
    assert np.all(results[:, 1] >= results[-1, 1])
