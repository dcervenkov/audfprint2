"""Unit tests for matcher utilities and match orchestration."""

import numpy as np

from audfprint2.core import analyzer, hash_table, matcher


class TestMatcherHelpers:
    def test_encpowerof2_handles_small_values(self):
        assert matcher.encpowerof2(0) == 0
        assert matcher.encpowerof2(3) == 2

    def test_keep_local_maxes_preserves_peak_values(self):
        vec = np.array([0, 2, 1, 3, 1])
        kept = matcher.keep_local_maxes(vec)

        assert kept[1] == 2
        assert kept[3] == 3
        # Non-peaks should be zeroed
        assert kept[2] == 0

    def test_find_modes_returns_local_maxima(self):
        data = np.array([0, 0, 1, 1, 1, 3, 3, 3])
        modes, counts = matcher.find_modes(data, threshold=1)

        assert list(modes) == [1, 3]
        assert list(counts) == [3, 3]


class TestMatchPipeline:
    def _create_hash_table(self):
        ht = hash_table.HashTable(hashbits=4, depth=5, maxtime=32)
        ht.store("alpha", [(0, 1), (1, 2), (2, 3)])
        ht.store("beta", [(0, 1), (10, 2)])
        return ht

    def test_unique_match_hashes_respects_window(self):
        hits = np.array([
            [0, 0, 1, 0],  # id, delta_time, hash, orig_time
            [0, 1, 1, 1],
            [0, 3, 2, 2],
        ])
        unique = matcher.Matcher()._unique_match_hashes(0, hits, mode=0)

        # Only hashes within the default window (1) around mode 0 are kept
        assert list(map(tuple, unique)) == [(0, 1), (1, 1)]

    def test_match_hashes_ranks_best_match_first(self):
        ht = self._create_hash_table()
        match = matcher.Matcher()
        match.threshcount = 1
        query = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32)

        results = match.match_hashes(ht, query)

        # alpha should be the top match with at least the query length in filtered counts
        assert results[0, 0] == 0
        assert results[0, 1] >= len(query)

    def test_match_file_with_precomputed_hashes(self, tmp_path):
        ht = self._create_hash_table()
        match = matcher.Matcher()
        match.threshcount = 1
        anlz = analyzer.Analyzer()

        precomputed = tmp_path / "query.afpt"
        hashes = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
        analyzer.hashes_save(precomputed, hashes)

        results, duration, nhashes = match.match_file(anlz, ht, str(precomputed))

        assert results.shape[0] == 1
        assert results[0, 0] == 0
        assert nhashes == len(hashes)
        # Duration is derived from the last hash time
        assert duration > 0
