import os
from pathlib import Path

import numpy as np
import pytest
import scipy.io.wavfile

from audfprint2 import cli
from audfprint2.core import analyzer, hash_table, matcher
from audfprint2.utils import audio


@pytest.fixture()
def temp_wav(tmp_path):
    sr = 11025
    t = np.linspace(0, 1.0, int(sr * 1.0), endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * 220 * t)
    mod = 0.25 * np.sin(2 * np.pi * 440 * t)
    data = (tone + mod)
    wav_path = tmp_path / "tone.wav"
    scipy.io.wavfile.write(wav_path, sr, np.int16(data * 32767))
    return wav_path


@pytest.fixture(autouse=True)
def disable_ffmpeg(monkeypatch):
    monkeypatch.setattr(audio, "HAVE_FFMPEG", False)


@pytest.fixture()
def sample_hashes():
    return np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32)


def test_audio_read_wav_and_mismatch(temp_wav, tmp_path):
    data, sr = audio.audio_read(str(temp_wav), sr=11025, channels=1)
    assert sr == 11025
    assert data.ndim == 1
    assert data.size > 0

    mismatch = tmp_path / "mismatch.wav"
    scipy.io.wavfile.write(mismatch, 8000, np.zeros(800, dtype=np.int16))
    with pytest.raises(ValueError):
        audio.audio_read(str(mismatch), sr=11025, channels=1)


@pytest.fixture()
def analyzer_obj():
    obj = analyzer.Analyzer(density=10.0)
    obj.fail_on_error = True
    return obj


@pytest.fixture()
def hash_tab():
    return hash_table.HashTable(hashbits=10, depth=2, maxtime=1 << 10)


def test_filename_list_iterator(tmp_path):
    list_file = tmp_path / "files.txt"
    list_file.write_text("tone\nother")
    iterator = cli.filename_list_iterator([str(list_file)], wavdir="/tmp", wavext=".wav", listflag=True)
    assert list(iterator) == ["/tmp/tone.wav", "/tmp/other.wav"]

    iterator = cli.filename_list_iterator(["single"], wavdir="/tmp", wavext=".wav", listflag=False)
    assert list(iterator) == ["/tmp/single.wav"]


def test_file_precompute_and_skip(analyzer_obj, temp_wav, tmp_path, sample_hashes, monkeypatch):
    monkeypatch.setattr(analyzer_obj, "wavfile2hashes", lambda filename: sample_hashes)
    precomp_dir = tmp_path / "pre"
    messages = cli.file_precompute_peaks_or_hashes(
        analyzer_obj, str(temp_wav), str(precomp_dir), hashes_not_peaks=True
    )
    assert precomp_dir.exists()
    assert "wrote" in messages[0]

    skip_messages = cli.file_precompute_peaks_or_hashes(
        analyzer_obj, str(temp_wav), str(precomp_dir), hashes_not_peaks=True, skip_existing=True
    )
    assert "skipping" in skip_messages[0]


def test_do_cmd_and_match(analyzer_obj, hash_tab, temp_wav, tmp_path, sample_hashes, monkeypatch):
    monkeypatch.setattr(analyzer_obj, "wavfile2hashes", lambda filename: sample_hashes)
    analyzer_obj.soundfiletotaldur = 1.0
    cli.do_cmd(
        "new",
        analyzer_obj,
        hash_tab,
        [str(temp_wav)],
        matcher_obj=None,
        outdir=str(tmp_path),
        type="hashes",
    )
    assert hash_tab.names == [str(temp_wav)]

    m = matcher.Matcher()
    cli.do_cmd(
        "match",
        analyzer_obj,
        hash_tab,
        [str(temp_wav)],
        matcher_obj=m,
        outdir=str(tmp_path),
        type="hashes",
    )

    with pytest.raises(ValueError):
        cli.do_cmd(
            "unknown",
            analyzer_obj,
            hash_tab,
            [],
            matcher_obj=m,
            outdir=str(tmp_path),
            type="hashes",
        )


def test_do_cmd_multiproc_unknown(analyzer_obj, hash_tab, temp_wav, tmp_path, sample_hashes, monkeypatch):
    monkeypatch.setattr(analyzer_obj, "wavfile2hashes", lambda filename: sample_hashes)
    with pytest.raises(ValueError):
        cli.do_cmd_multiproc(
            "remove",
            analyzer_obj,
            hash_tab,
            [str(temp_wav)],
            matcher_obj=None,
            outdir=str(tmp_path),
            type="hashes",
        )


@pytest.mark.parametrize("value,expected", [(1, 0), (2, 1), (3, 2), (4, 2)])
def test_encpowerof2(value, expected):
    assert matcher.encpowerof2(value) == expected


def test_matcher_unique_hashes_and_time_ranges(hash_tab, analyzer_obj, temp_wav, sample_hashes, monkeypatch):
    monkeypatch.setattr(analyzer_obj, "wavfile2hashes", lambda filename: sample_hashes)
    hash_tab.store(str(temp_wav), sample_hashes)
    hash_tab.hashesperid = np.array([len(sample_hashes)])

    hit_matrix = []
    for idx, (time_bin, hash_val) in enumerate(sample_hashes):
        hit_matrix.append([0, int(time_bin), int(hash_val), int(idx)])
    hits = np.array(hit_matrix, dtype=int)

    m = matcher.Matcher()
    m.threshcount = 1
    ids, counts = m._best_count_ids(hits, hash_tab)
    assert ids[0] == 0
    assert counts[0] >= len(hits)

    unique = m._unique_match_hashes(0, hits, mode=int(hits[0, 1]))
    assert unique.shape[1] == 2

    min_time, max_time = m._calculate_time_ranges(hits, 0, mode=int(hits[0, 1]))
    assert min_time <= max_time
