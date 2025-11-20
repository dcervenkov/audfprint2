import math
from pathlib import Path

import pytest

from audfprint2 import cli


def test_main_requires_dbase_for_non_precompute():
    with pytest.raises(ValueError):
        cli.main(["new", "file.wav"])


def test_main_precompute_multiproc_dispatch(monkeypatch, tmp_path):
    called = {}

    def fake_do_cmd_multiproc(cmd, analyzer_obj, hash_tab, filename_iter, matcher_obj, outdir, type, **kwargs):
        called["cmd"] = cmd
        called["files"] = list(filename_iter)
        called["skip_existing"] = kwargs.get("skip_existing")
        called["strip_prefix"] = kwargs.get("strip_prefix")

    monkeypatch.setattr(cli, "do_cmd_multiproc", fake_do_cmd_multiproc)

    cli.main([
        "precompute",
        "--ncores",
        "2",
        "--wavdir",
        str(tmp_path),
        "--wavext",
        ".wav",
        "song",
    ])

    assert called["cmd"] == "precompute"
    assert called["files"] == [str(tmp_path / "song.wav")]
    assert called["strip_prefix"] == str(tmp_path)
    assert called["skip_existing"] is False


def test_main_new_creates_hash_table_and_saves(monkeypatch, tmp_path):
    created = {}

    class DummyHashTable:
        def __init__(self, hashbits, depth, maxtime):
            created["init"] = (hashbits, depth, maxtime)
            self.hashbits = hashbits
            self.depth = depth
            self.maxtimebits = int(math.log2(maxtime))
            self.params = {}
            self.dirty = False

        def save(self, path):
            created["saved"] = path

    def fake_do_cmd(cmd, analyzer_obj, hash_tab, filename_iter, matcher_obj, outdir, type, **kwargs):
        created["cmd"] = cmd
        created["files"] = list(filename_iter)
        hash_tab.dirty = True  # force save branch

    monkeypatch.setattr(cli.hash_table, "HashTable", DummyHashTable)
    monkeypatch.setattr(cli, "do_cmd", fake_do_cmd)

    dbase_path = tmp_path / "dbase.pkl"
    cli.main([
        "new",
        "--dbase",
        str(dbase_path),
        "--hashbits",
        "8",
        "--bucketsize",
        "3",
        "--maxtimebits",
        "4",
        "--wavdir",
        str(tmp_path),
        "--wavext",
        ".wav",
        "clip",
    ])

    assert created["cmd"] == "new"
    assert created["files"] == [str(tmp_path / "clip.wav")]
    assert created["init"] == (8, 3, 16)
    assert created["saved"] == str(dbase_path)


def test_do_cmd_merge_and_remove(tmp_path):
    base = cli.hash_table.HashTable(hashbits=4, depth=2, maxtime=32)
    base.params["samplerate"] = 11025
    base.store("a", [(0, 1)])

    other = cli.hash_table.HashTable(hashbits=4, depth=2, maxtime=32)
    other.params["samplerate"] = 11025
    other.store("b", [(1, 2)])
    merged_path = tmp_path / "merged.pkl"
    other.save(str(merged_path))

    cli.do_cmd("merge", None, base, [str(merged_path)], None, outdir=str(tmp_path), type="hashes")

    assert set(filter(None, base.names)) == {"a", "b"}

    cli.do_cmd("remove", None, base, ["a"], None, outdir=str(tmp_path), type="hashes")
    assert "a" not in base.names
