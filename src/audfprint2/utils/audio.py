# coding=utf-8
"""audio_read reads in a whole audio file with resampling."""

# Equivalent to:
# import librosa
# def audio_read(filename, sr=11025, channels=1):
#    """Read a soundfile, return (d, sr)."""
#    d, sr = librosa.load(filename, sr=sr, mono=(channels == 1))
#    return d, sr

# The code below is adapted from:
# https://github.com/bmcfee/librosa/blob/master/librosa/core/audio.py
# This is its original copyright notice:

# Copyright (c) 2014, Brian McFee, Matt McVicar, Dawen Liang, Colin Raffel, Douglas Repetto, Dan Ellis.
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
# OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
# PERFORMANCE OF THIS SOFTWARE.

import os
import queue
import re
import subprocess
import threading
import time
from collections.abc import Iterator
from types import TracebackType
from typing import IO, Literal

import numpy as np

# For wavread fallback.
import scipy.io.wavfile as wav  # type: ignore[import-untyped]

# If ffmpeg is unavailable, you can set HAVE_FFMPEG to False which will cause
# soundfile reads to go via scipy.io.wavfile.  However, this means that only
# *.wav files are supported *and* they must already be resampled to the
# system sampling rate (e.g. 11025 Hz).

HAVE_FFMPEG = True


def wavread(filename: str) -> tuple[np.ndarray, int]:
    """Read in audio data from a wav file.  Return d, sr."""
    # Read in wav file.
    samplerate, wave_data = wav.read(filename)
    # Normalize short ints to floats in range [-1..1).
    data = np.asarray(wave_data, dtype=np.float64) / 32768.0
    return data, samplerate


def audio_read(
    filename: str,
    sr: int | None = None,
    channels: int | None = None,
) -> tuple[np.ndarray, int]:
    """Read a soundfile, return (d, sr)."""
    if HAVE_FFMPEG:
        return audio_read_ffmpeg(filename, sr, channels)
    data, samplerate = wavread(filename)
    if channels == 1 and len(data.shape) == 2 and data.shape[-1] != 1:
        # Convert stereo to mono.
        data = np.mean(data, axis=-1)
    if sr and sr != samplerate:
        raise ValueError("Wav file has samplerate %f but %f requested." % (samplerate, sr))
    return data, samplerate


def audio_read_ffmpeg(
    filename: str,
    sr: int | None = None,
    channels: int | None = None,
) -> tuple[np.ndarray, int]:
    """Read a soundfile, return (d, sr)."""
    # Hacked version of librosa.load and audioread/ff.
    offset = 0.0
    duration = None
    dtype = np.float32
    frames: list[np.ndarray] = []
    with FFmpegAudioFile(os.path.realpath(filename), sample_rate=sr, channels=channels) as input_file:
        sr = input_file.sample_rate
        channels = input_file.channels
        if sr is None or channels is None:
            raise ValueError("Sample rate or channels could not be determined.")
        s_start = int(np.floor(sr * offset) * channels)
        if duration is None:
            s_end = np.inf
        else:
            s_end = s_start + int(np.ceil(sr * duration) * channels)
        num_read = 0
        for frame_bytes in input_file:
            frame = buf_to_float(frame_bytes, dtype=dtype)
            num_read_prev = num_read
            num_read += len(frame)
            if num_read < s_start:
                # offset is after the current frame, keep reading.
                continue
            if s_end < num_read_prev:
                # we're off the end.  stop reading
                break
            if s_end < num_read:
                # the end is in this frame.  crop.
                frame = frame[: int(s_end - num_read_prev)]
            if num_read_prev <= s_start < num_read:
                # beginning is in this frame
                frame = frame[(s_start - num_read_prev) :]
            # tack on the current frame
            frames.append(frame)

        if not len(frames):
            # Zero-length read
            y: np.ndarray = np.zeros(0, dtype=dtype)
        else:
            y = np.concatenate(frames)
            if channels > 1:
                y = y.reshape((-1, 2)).T

    # Final cleanup for dtype and contiguity
    y = np.ascontiguousarray(y, dtype=dtype)

    return (y, sr)


def buf_to_float(
    x: bytes | bytearray | memoryview | np.ndarray,
    n_bytes: int = 2,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Convert an integer buffer to floating point values.
    This is primarily useful when loading integer-valued wav data
    into numpy arrays.
    .. seealso:: :func:`librosa.util.buf_to_float`
    :parameters:
        - x : np.ndarray [dtype=int]
            The integer-valued data buffer
        - n_bytes : int [1, 2, 4]
            The number of bytes per sample in ``x``
        - dtype : numeric type
            The target output type (default: 32-bit float)
    :return:
        - x_float : np.ndarray [dtype=float]
            The input data buffer cast to floating point
    """

    # Invert the scale of the data
    scale = 1.0 / float(1 << ((8 * n_bytes) - 1))

    # Construct the format string
    fmt = "<i{:d}".format(n_bytes)

    # Rescale and format the data buffer
    return scale * np.frombuffer(x, fmt).astype(dtype)


# The code below is adapted from:
# https://github.com/sampsyo/audioread/blob/master/audioread/ffdec.py
# Below is its original copyright notice:

# This file is part of audioread.
# Copyright 2014, Adrian Sampson.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.


class QueueReaderThread(threading.Thread):
    """A thread that consumes data from a filehandle and sends the data
    over a Queue.
    """

    def __init__(self, fh: IO[bytes], blocksize: int = 1024, discard: bool = False) -> None:
        super(QueueReaderThread, self).__init__()
        self.fh = fh
        self.blocksize = blocksize
        self.daemon = True
        self.discard = discard
        self.queue: queue.Queue[bytes] = queue.Queue()

    def run(self) -> None:
        while True:
            data = self.fh.read(self.blocksize)
            if not self.discard:
                self.queue.put(data)
            if not data:
                # Stream closed (EOF).
                break


class FFmpegAudioFile(object):
    """An audio file decoded by the ffmpeg command-line utility."""

    def __init__(
        self,
        filename: str,
        channels: int | None = None,
        sample_rate: int | None = None,
        block_size: int = 4096,
    ) -> None:
        if not os.path.isfile(filename):
            raise ValueError(f"{filename} not found.")
        popen_args = ["ffmpeg", "-i", filename, "-f", "s16le"]
        self.channels = channels
        self.sample_rate = sample_rate
        if channels:
            popen_args.extend(["-ac", str(channels)])
        if sample_rate:
            popen_args.extend(["-ar", str(sample_rate)])
        popen_args.append("-")
        self.proc = subprocess.Popen(popen_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Start another thread to consume the standard output of the
        # process, which contains raw audio data.
        if self.proc.stdout is None:
            raise ValueError("stdout stream unavailable")
        self.stdout_reader = QueueReaderThread(self.proc.stdout, block_size)
        self.stdout_reader.start()

        # Read relevant information from stderr.
        try:
            self._get_info()
        except ValueError as err:
            raise ValueError(f"Error reading header info from {filename}") from err

        # Start a separate thread to read the rest of the data from
        # stderr. This (a) avoids filling up the OS buffer and (b)
        # collects the error output for diagnosis.
        if self.proc.stderr is None:
            raise ValueError("stderr stream unavailable")
        self.stderr_reader = QueueReaderThread(self.proc.stderr)
        self.stderr_reader.start()

    def read_data(self, timeout: float = 10.0) -> Iterator[bytes]:
        """Read blocks of raw PCM data from the file."""
        # Read from stdout in a separate thread and consume data from
        # the queue.
        start_time = time.time()
        while True:
            # Wait for data to be available or a timeout.
            data = None
            try:
                data = self.stdout_reader.queue.get(timeout=timeout)
                if data:
                    yield data
                else:
                    # End of file.
                    break
            except queue.Empty as err:
                # Queue read timed out.
                end_time = time.time()
                if not data:
                    if end_time - start_time >= timeout:
                        # Nothing interesting has happened for a while --
                        # FFmpeg is probably hanging.
                        raise ValueError(
                            f'ffmpeg output: {b"".join(self.stderr_reader.queue.queue).decode("utf8", "ignore")}'
                        ) from err
                    start_time = end_time
                    # Keep waiting.
                    continue

    def _get_info(self) -> None:
        """Reads the tool's output from its stderr stream, extracts the
        relevant information, and parses it.
        """
        if self.proc.stderr is None:
            raise ValueError("stderr stream unavailable")
        stderr = self.proc.stderr
        out_parts = []
        while True:
            line_bytes = stderr.readline()
            if not line_bytes:
                # EOF and data not found.
                raise ValueError("stream info not found")

            line = line_bytes.decode("utf8", "ignore")

            line = line.strip().lower()

            if "no such file" in line:
                raise IOError("file not found")
            elif "invalid data found" in line:
                raise ValueError()
            elif "duration:" in line:
                out_parts.append(line)
            elif "audio:" in line:
                out_parts.append(line)
                self._parse_info("".join(out_parts))
                break

    def _parse_info(self, s: str) -> None:
        """Given relevant data from the ffmpeg output, set audio
        parameter fields on this object.
        """
        if match := re.search(r"(\d+) hz", s):
            self.sample_rate_orig = int(match[1])
        else:
            self.sample_rate_orig = 0
        if self.sample_rate is None:
            self.sample_rate = self.sample_rate_orig

        if match := re.search(r"hz, ([^,]+),", s):
            mode = match[1]
            if mode == "stereo":
                self.channels_orig = 2
            elif match := re.match(r"(\d+) ", mode):
                self.channels_orig = int(match[1])
            else:
                self.channels_orig = 1
        else:
            self.channels_orig = 0
        if self.channels is None:
            self.channels = self.channels_orig

        if match := re.search(r"duration: (\d+):(\d+):(\d+).(\d)", s):
            durparts = list(map(int, match.groups()))
            duration = durparts[0] * 60 * 60 + durparts[1] * 60 + durparts[2] + float(durparts[3]) / 10
            self.duration = duration
        else:
            # No duration found.
            self.duration = 0

    def close(self) -> None:
        """Close the ffmpeg process used to perform the decoding."""
        # Kill the process if it is still running.
        if hasattr(self, "proc") and self.proc.returncode is None:
            self.proc.kill()
            self.proc.wait()

    def __del__(self) -> None:
        self.close()

    # Iteration.
    def __iter__(self) -> Iterator[bytes]:
        return self.read_data()

    # Context manager.
    def __enter__(self) -> "FFmpegAudioFile":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        self.close()
        return False
