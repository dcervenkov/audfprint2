"""Unit tests for the HashTable implementation."""

import numpy as np
import pytest

from audfprint2.core.hash_table import HashTable, _bitsfor


class TestBitsfor:
    def test_bitsfor_power_of_two(self):
        assert _bitsfor(16) == 4
        assert _bitsfor(1024) == 10

    def test_bitsfor_raises_on_non_power_of_two(self):
        with pytest.raises(ValueError):
            _bitsfor(12)


class TestHashTableStorage:
    def test_store_and_retrieve_round_trip(self):
        ht = HashTable(hashbits=8, depth=5, maxtime=64)
        pairs = [(1, 3), (10, 15), (20, 31)]

        ht.store("track", pairs)

        retrieved = ht.retrieve("track")
        # retrieved order depends on hash buckets; compare sorted rows
        np.testing.assert_array_equal(np.sort(retrieved, axis=0), np.sort(np.array(pairs), axis=0))
        assert ht.names[0] == "track"

    def test_get_hits_matches_stored_hashes(self):
        ht = HashTable(hashbits=6, depth=4, maxtime=64)
        pairs = [(5, 8), (6, 8)]
        ht.store("song", pairs)

        hits = ht.get_hits(np.array(pairs, dtype=np.int32))
        # Two stored hashes for the same bucket should yield four hits (two per query row)
        assert hits.shape == (4, 4)

        # All hits should reference the same file id and hash bucket
        assert np.all(hits[:, 0] == 0)
        assert np.all(hits[:, 2] == 8)

        # Delta times reflect stored time minus query time for each lookup
        expected_deltas = {0, 1, -1}
        assert set(hits[:, 1]) <= expected_deltas

    def test_remove_clears_entries_for_name(self):
        ht = HashTable(hashbits=6, depth=4, maxtime=64)
        pairs_a = [(0, 1), (1, 2)]
        pairs_b = [(0, 1)]
        ht.store("a", pairs_a)
        ht.store("b", pairs_b)

        ht.remove("a")

        assert ht.names[0] is None
        assert ht.hashesperid[0] == 0
        remaining_entries = ht.retrieve("b")
        np.testing.assert_array_equal(remaining_entries[:, 1], np.array([1], dtype=np.int32))


class TestHashTablePersistence:
    def test_reset_clears_data_but_preserves_shape(self):
        ht = HashTable(hashbits=4, depth=3, maxtime=16)
        ht.store("x", [(1, 2), (2, 3)])

        ht.reset()

        assert len(ht.names) == 0
        assert ht.table.shape == (2 ** ht.hashbits, ht.depth)
        assert np.all(ht.table == 0)

    @pytest.mark.parametrize("save_type", ["PKL", "HDF"])
    def test_save_and_reload_round_trip(self, tmp_path, save_type):
        ht = HashTable(hashbits=4, depth=4, maxtime=16)
        pairs = [(0, 3), (1, 4), (2, 5)]
        ht.store("original", pairs)
        target = tmp_path / f"table.{save_type.lower()}"

        ht.save(str(target), save_type=save_type)

        loaded = HashTable(str(target))

        assert loaded.names[0] == "original"
        np.testing.assert_array_equal(loaded.retrieve("original")[:, 1], np.array([3, 4, 5], dtype=np.int32))
        assert not loaded.dirty
