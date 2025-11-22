# coding=utf-8
"""
audfprint2_match.py

Fingerprint matching code for audfprint2

2014-05-26 Dan Ellis dpwe@ee.columbia.edu
"""

import os
import time
from typing import cast

import numpy as np
import psutil  # type: ignore[import-untyped]
import scipy.signal  # type: ignore[import-untyped]

import logging

from audfprint2.core import analyzer, hash_table
from audfprint2.utils import audio, stft

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("audfprint2")
TRACE_LEVEL = logging.DEBUG - 5


def process_info() -> tuple[int, float]:
    rss = usrtime = 0
    p = psutil.Process(os.getpid())
    if os.name == 'nt':
        rss = p.memory_info()[0]
        usrtime = p.cpu_times()[0]
    else:
        rss = p.get_memory_info()[0]
        usrtime = p.get_cpu_times()[0]
    return rss, usrtime


def log(message: str) -> None:
    """ log info with stats """
    rss, usrtime = process_info()
    logger.debug(f'{time.ctime()} physmem={rss} utime={usrtime} {message}')


def encpowerof2(val: int) -> int:
    """ Return N s.t. 2^N >= val """
    return int(np.ceil(np.log(max(1, val)) / np.log(2)))


def locmax(vec: np.ndarray, indices: bool = False) -> np.ndarray:
    """ Return a boolean vector of which points in vec are local maxima.
        End points are peaks if larger than single neighbors.
        if indices=True, return the indices of the True values instead
        of the boolean vector. (originally from audfprint2.py)
    """
    # x[-1]-1 means last value can be a peak
    # nbr = np.greater_equal(np.r_[x, x[-1]-1], np.r_[x[0], x])
    # the np.r_ was killing us, so try an optimization...
    nbr = np.zeros(len(vec) + 1, dtype=bool)
    nbr[0] = True
    nbr[1:-1] = np.greater_equal(vec[1:], vec[:-1])
    maxmask = (nbr[:-1] & ~nbr[1:])
    return np.nonzero(maxmask)[0] if indices else maxmask


def keep_local_maxes(vec: np.ndarray) -> np.ndarray:
    """ Zero out values unless they are local maxima."""
    local_maxes = np.zeros(vec.shape)
    locmaxindices = locmax(vec, indices=True)
    local_maxes[locmaxindices] = vec[locmaxindices]
    return local_maxes


def find_modes(
    data: np.ndarray,
    threshold: int = 5,
    window: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """ Find multiple modes in data,  Report a list of (mode, count)
        pairs for every mode greater than or equal to threshold.
        Only local maxima in counts are returned.
    """
    # TODO: Ignores window at present
    datamin = np.amin(data)
    fullvector = np.bincount(data - datamin)
    # Find local maxima
    localmaxes = np.nonzero(np.logical_and(locmax(fullvector),
                                           np.greater_equal(fullvector,
                                                            threshold)))[0]
    return localmaxes + datamin, fullvector[localmaxes]


class Matcher(object):
    """Provide matching for audfprint2 fingerprint queries to hash table"""

    def __init__(self) -> None:
        """Set up default object values"""
        # Tolerance window for time differences
        self.window = 1
        # Absolute minimum number of matching hashes to count as a match
        self.threshcount = 5
        # How many hits to return?
        self.max_returns = 1
        # How deep to search in return list?
        self.search_depth = 100
        # Sort those returns by time (instead of counts)?
        self.sort_by_time = False
        # Do illustration?
        self.illustrate = False
        # Careful counts?
        self.exact_count = False
        # Search for time range?
        self.find_time_range = False
        # Quantile of time range to report.
        self.time_quantile = 0.02
        # Display pre-emphasized spectrogram in illustrate_match?
        self.illustrate_hpf = False
        # If there are a lot of matches within a single track at different
        # alignments, stop looking after a while.
        self.max_alignments_per_id = 100

    def _best_count_ids(
        self,
        hits: np.ndarray,
        ht: hash_table.HashTable,
    ) -> tuple[np.ndarray, np.ndarray]:
        """ Return the indexes for the ids with the best counts.
            hits is a matrix as returned by hash_table.get_hits()
            with rows of consisting of [id dtime hash otime] """
        allids = hits[:, 0]
        ids = np.unique(allids)
        # rawcounts = np.sum(np.equal.outer(ids, allids), axis=1)
        # much faster, and doesn't explode memory
        rawcounts = np.bincount(allids)[ids]
        # Divide the raw counts by the total number of hashes stored
        # for the ref track, to downweight large numbers of chance
        # matches against longer reference tracks.
        wtdcounts = rawcounts / (ht.hashesperid[ids].astype(float))

        # Find all the actual hits for a the most popular ids
        bestcountsixs = np.argsort(wtdcounts)[::-1]
        # We will examine however many hits have rawcounts above threshold
        # up to a maximum of search_depth.
        maxdepth = np.minimum(np.count_nonzero(np.greater(rawcounts,
                                                          self.threshcount)),
                              self.search_depth)
        # Return the ids to check
        bestcountsixs = bestcountsixs[:maxdepth]
        return ids[bestcountsixs], rawcounts[bestcountsixs]

    def _unique_match_hashes(self, id: int, hits: np.ndarray, mode: int) -> np.ndarray:
        """ Return the list of unique matching hashes.  Split out so
            we can recover the actual matching hashes for the best
            match if required. """
        allids = hits[:, 0]
        alltimes = hits[:, 1]
        allhashes = hits[:, 2].astype(np.int64)
        allotimes = hits[:, 3]
        timebits = max(1, encpowerof2(np.amax(allotimes)))
        # matchhashes may include repeats because multiple
        # ref hashes may match a single query hash under window.
        # Uniqify:
        # matchhashes = sorted(list(set(matchhashes)))
        # much, much faster:
        matchix = np.nonzero(
                np.logical_and(allids == id, np.less_equal(np.abs(alltimes - mode),
                                                           self.window)))[0]
        matchhasheshash = np.unique(allotimes[matchix]
                                    + (allhashes[matchix] << timebits))
        timemask = (1 << timebits) - 1
        return np.c_[matchhasheshash & timemask, matchhasheshash >> timebits]

    def _calculate_time_ranges(
        self,
        hits: np.ndarray,
        id: int,
        mode: int,
    ) -> tuple[int, int]:
        """Given the id and mode, return the actual time support.
           hits is an np.array of id, skew_time, hash, orig_time
           which must be sorted in orig_time order."""
        minoffset = mode - self.window
        maxoffset = mode + self.window
        # match_times = sorted(hits[row, 3]
        #                     for row in np.nonzero(hits[:, 0]==id)[0]
        #                     if mode - self.window <= hits[row, 1]
        #                     and hits[row, 1] <= mode + self.window)
        match_times = hits[np.logical_and.reduce([
            hits[:, 1] >= minoffset,
            hits[:, 1] <= maxoffset,
            hits[:, 0] == id
        ]), 3]
        min_time = match_times[int(len(match_times) * self.time_quantile)]
        max_time = match_times[int(len(match_times) * (1.0 - self.time_quantile)) - 1]
        # log("_calc_time_ranges: len(hits)={:d} id={:d} mode={:d} matches={:d} min={:d} max={:d}".format(
        #    len(hits), id, mode, np.sum(np.logical_and(hits[:, 1] >= minoffset,
        #                                               hits[:, 1] <= maxoffset)),
        #    min_time, max_time))
        return min_time, max_time

    def _exact_match_counts(
        self,
        hits: np.ndarray,
        ids: np.ndarray,
        rawcounts: np.ndarray,
        hashesfor: int | None = None,
    ) -> np.ndarray:
        """Find the number of "filtered" (time-consistent) matching hashes
            for each of the promising ids in <ids>.  Return an
            np.array whose rows are [id, filtered_count,
            modal_time_skew, unfiltered_count, original_rank,
            min_time, max_time].  Results are sorted by original rank
            (but will not in general include all the the original
            IDs).  There can be multiple rows for a single ID, if
            there are several distinct time_skews giving good
            matches.
        """
        # Sort hits into time_in_original order - needed for _calc_time_range
        sorted_hits = hits[hits[:, 3].argsort()]
        # Slower, old process for exact match counts
        allids = sorted_hits[:, 0]
        alltimes = sorted_hits[:, 1]
        # allhashes = sorted_hits[:, 2]
        # allotimes = sorted_hits[:, 3]
        # Allocate enough space initially for 4 modes per hit
        maxnresults = len(ids) * 4
        results = np.zeros((maxnresults, 7), np.int32)
        nresults = 0
        min_time = 0
        max_time = 0
        for urank, (id, rawcount) in enumerate(zip(ids, rawcounts, strict=False)):
            modes, counts = find_modes(alltimes[np.nonzero(allids == id)[0]],
                                       window=self.window,
                                       threshold=self.threshcount)
            for mode in modes:
                matchhashes = self._unique_match_hashes(id, sorted_hits, mode)
                # Now we get the exact count
                filtcount = len(matchhashes)
                if filtcount >= self.threshcount:
                    if nresults == maxnresults:
                        # Extend array
                        maxnresults *= 2
                        results.resize((maxnresults, results.shape[1]))
                    if self.find_time_range:
                        min_time, max_time = self._calculate_time_ranges(
                                sorted_hits, id, mode)
                    results[nresults, :] = [id, filtcount, mode, rawcount,
                                            urank, min_time, max_time]
                    nresults += 1
        return results[:nresults, :]

    def _approx_match_counts(
        self,
        hits: np.ndarray,
        ids: np.ndarray,
        rawcounts: np.ndarray,
    ) -> np.ndarray:
        """ Quick and slightly inaccurate routine to count time-aligned hits.

        Only considers largest mode for reference ID match.

        Args:
          hits: np.array of hash matches, each row consists of
            <track_id, skew_time, hash, orig_time>.
          ids: list of the IDs to check, based on raw match count.
          rawcounts: list giving the actual raw counts for each id to try.

        Returns:
            Rows of [id, filt_count, time_skew, raw_count, orig_rank,
            min_time, max_time].
            Ids occur in the same order as the input list, but ordering
            of (potentially multiple) hits within each track may not be
            sorted (they are sorted by the largest single count value, not
            the total count integrated over -window:+window bins).
        """
        # In fact, the counts should be the same as exact_match_counts
        # *but* some matches may be pruned because we don't bother to
        # apply the window (allowable drift in time alignment) unless
        # there are more than threshcount matches at the single best time skew.
        # Note: now we allow multiple matches per ID, this may need to grow
        # so it can grow inside the loop.
        results = np.zeros((len(ids), 7), np.int32)
        if not hits.size:
            # No hits found, return empty results
            return results
        # Sort hits into time_in_original order - needed for _calc_time_range
        sorted_hits = hits[hits[:, 3].argsort()]
        allids = sorted_hits[:, 0].astype(int)
        alltimes = sorted_hits[:, 1].astype(int)
        window = int(self.window)
        # Make sure every value in alltimes is >=0 for bincount
        mintime = np.amin(alltimes)
        alltimes -= mintime
        nresults = 0
        min_time = 0
        max_time = 0
        for urank, (track_id, rawcount) in enumerate(zip(ids, rawcounts, strict=False)):
            # Make sure id is an int64 before shifting it up.
            track_id = int(track_id)
            # Select the subrange of bincounts corresponding to this id
            bincounts = np.bincount(alltimes[allids == track_id])
            still_looking = True
            # Only consider legit local maxima in bincounts.
            filtered_bincounts = keep_local_maxes(bincounts)
            found_this_id = 0
            while still_looking:
                mode = int(np.argmax(filtered_bincounts))
                if filtered_bincounts[mode] <= self.threshcount:
                    # Too few - skip to the next id
                    still_looking = False
                    continue
                start = max(0, mode - window)
                stop = mode + window + 1
                count = np.sum(bincounts[start:stop])
                if self.find_time_range:
                    min_time, max_time = self._calculate_time_ranges(
                            sorted_hits, track_id, mode + mintime)
                results[nresults, :] = [track_id, count, mode + mintime, rawcount,
                                        urank, min_time, max_time]
                nresults += 1
                if nresults >= results.shape[0]:
                    results = np.vstack([results, np.zeros(results.shape,
                                                           np.int32)])
                # Clear this hit to find next largest.
                filtered_bincounts[max(0, mode - self.window):
                                   (mode + self.window + 1)] = 0
                found_this_id += 1
                if found_this_id > self.max_alignments_per_id:
                    still_looking = False
        return results[:nresults, :]

    def match_hashes(
        self,
        ht: hash_table.HashTable,
        hashes: np.ndarray,
        hashesfor: int | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """ Match audio against fingerprint hash table.
            Return top N matches as (id, filteredmatches, timoffs, rawmatches,
            origrank, mintime, maxtime)
            If hashesfor specified, return the actual matching hashes for that
            hit (0=top hit).
        """
        # find the implicated id, time pairs from hash table
        # log("nhashes=%d" % np.shape(hashes)[0])
        hits = ht.get_hits(hashes)

        bestids, rawcounts = self._best_count_ids(hits, ht)

        # log("len(rawcounts)=%d max(rawcounts)=%d" %
        #    (len(rawcounts), max(rawcounts)))
        if not self.exact_count:
            results = self._approx_match_counts(hits, bestids, rawcounts)
        else:
            results = self._exact_match_counts(hits, bestids, rawcounts,
                                               hashesfor)
        # Sort results by filtered count, descending
        results = results[(-results[:, 1]).argsort(),]
        if hashesfor is None:
            return results
        track_id = results[hashesfor, 0]
        mode = results[hashesfor, 2]
        hashesforhashes = self._unique_match_hashes(track_id, hits, mode)
        return results, hashesforhashes

    def match_file(
        self,
        analyzer_obj: analyzer.Analyzer,
        ht: hash_table.HashTable,
        filename: str,
        number: int | None = None,
    ) -> tuple[np.ndarray, float, int]:
        """ Read in an audio file, calculate its landmarks, query against
            hash table.  Return top N matches as (id, filterdmatchcount,
            timeoffs, rawmatchcount), also length of input file in sec,
            and count of raw query hashes extracted
        """
        q_hashes = analyzer_obj.wavfile2hashes(filename)
        # Fake durations as largest hash time
        if len(q_hashes) == 0:
            durd = 0.0
        else:
            durd = analyzer_obj.n_hop * q_hashes[-1][0] / analyzer_obj.target_sr

        numberstring = "#%d" % number if number is not None else ""
        logger.debug("%s Analyzed %s %s of %.3f s to %d hashes",
                     time.ctime(), numberstring, filename, durd, len(q_hashes))

        # Run query
        rslts = cast(np.ndarray, self.match_hashes(ht, q_hashes))
        # Post filtering
        if self.sort_by_time:
            rslts = rslts[(-rslts[:, 2]).argsort(), :]
        return rslts[:self.max_returns, :], durd, len(q_hashes)

    def file_match_to_msgs(
        self,
        analyzer_obj: analyzer.Analyzer,
        ht: hash_table.HashTable,
        qry: str,
        number: int | None = None,
    ) -> list[str]:
        """ Perform a match on a single input file, return list
            of message strings """
        rslts, dur, nhash = self.match_file(analyzer_obj, ht, qry, number)
        t_hop = analyzer_obj.n_hop / analyzer_obj.target_sr
        show_verbose = logger.isEnabledFor(logging.DEBUG)
        qrymsg = f"{qry} {dur:.1f} sec {nhash} raw hashes" if show_verbose else qry
        msgrslt = []
        if len(rslts) == 0:
            # No matches returned at all
            nhashaligned = 0
            msgrslt.append(f"NOMATCH: {qrymsg}")
        else:
            for (tophitid, nhashaligned, aligntime, nhashraw, rank,
                 min_time, max_time) in rslts:
                # figure the number of raw and aligned matches for top hit
                if show_verbose:
                    if self.find_time_range:
                        msg = (
                            f"Matched {(max_time - min_time) * t_hop:6.1f} s "
                            f"starting at {min_time * t_hop:6.1f} s in {qry}"
                            f" to time {(min_time + aligntime) * t_hop:6.1f} s "
                            f"in {str(ht.names[tophitid])}"
                        )
                    else:
                        msg = f"Matched {qrymsg} as {str(ht.names[tophitid])} at {aligntime * t_hop:6.1f} s"
                    msg += f" with {nhashaligned:5d} of {nhashraw:5d} common hashes at rank {rank:2d}"
                    msgrslt.append(msg)
                else:
                    msgrslt.append(f"{qrymsg}\t{ht.names[tophitid][:]}")
                if self.illustrate:
                    self.illustrate_match(analyzer_obj, ht, qry)
        return msgrslt

    def illustrate_match(
        self,
        analyzer_obj: analyzer.Analyzer,
        ht: hash_table.HashTable,
        filename: str,
    ) -> np.ndarray:
        """ Show the query fingerprints and the matching ones
            plotted over a spectrogram """
        # Make the spectrogram
        # d, sr = librosa.load(filename, sr=analyzer.target_sr)
        d, sr = audio.audio_read(filename, sr=analyzer_obj.target_sr, channels=1)
        sgram = np.abs(stft.stft(d, n_fft=analyzer_obj.n_fft,
                                 hop_length=analyzer_obj.n_hop,
                                 window=np.hanning(analyzer_obj.n_fft + 2)[1:-1]))
        sgram = 20.0 * np.log10(np.maximum(sgram, np.max(sgram) / 1e6))
        sgram = sgram - np.mean(sgram)
        # High-pass filter onset emphasis
        # [:-1,] discards top bin (nyquist) of sgram so bins fit in 8 bits
        # spectrogram enhancement
        if self.illustrate_hpf:
            HPF_POLE = 0.98
            sgram = np.array([scipy.signal.lfilter([1, -1],
                                                   [1, -HPF_POLE], s_row)
                              for s_row in sgram])[:-1, ]
        sgram = sgram - np.max(sgram)
        try:
            import librosa.display
            import matplotlib.pyplot as plt
        except Exception as exc:
            raise RuntimeError(
                "Plotting requires optional dependencies; install with "
                "`pip install audfprint2[plot]`."
            ) from exc
        librosa.display.specshow(sgram, sr=sr, hop_length=analyzer_obj.n_hop,
                                 y_axis='linear', x_axis='time',
                                 cmap='gray_r', vmin=-80.0, vmax=0)
        # Do the match?
        q_hashes = analyzer_obj.wavfile2hashes(filename)
        # Run query, get back the hashes for match zero
        results, matchhashes = self.match_hashes(ht, q_hashes, hashesfor=0)
        if self.sort_by_time:
            results = sorted(results, key=lambda x: -x[2])
        # Convert the hashes to landmarks
        q_hashes_list = [(int(time_), int(hash_)) for time_, hash_ in q_hashes]
        match_hashes_list = [(int(time_), int(hash_)) for time_, hash_ in matchhashes]
        lms = analyzer.hashes2landmarks(q_hashes_list)
        mlms = analyzer.hashes2landmarks(match_hashes_list)
        # Overplot on the spectrogram
        time_scale = analyzer_obj.n_hop / float(sr)
        freq_scale = float(sr)/analyzer_obj.n_fft
        plt.plot(time_scale * np.array([[x[0], x[0] + x[3]] for x in lms]).T,
                 freq_scale * np.array([[x[1], x[2]] for x in lms]).T,
                 '.-g')
        plt.plot(time_scale * np.array([[x[0], x[0] + x[3]] for x in mlms]).T,
                 freq_scale * np.array([[x[1], x[2]] for x in mlms]).T,
                 '.-r')
        # Add title
        plt.title(
            (
                f"{filename} : Matched as {ht.names[results[0][0]]}"
                + " with %d of %d hashes" % (len(matchhashes), len(q_hashes))
            )
        )
        # Display
        plt.show()
        # Return
        return results


def localtest() -> None:
    """Function to provide quick test"""
    pat = '/Users/dpwe/projects/shazam/Nine_Lives/*mp3'
    qry = 'query.mp3'
    hash_tab = analyzer.glob2hashtable(pat)
    matcher = Matcher()
    g2h = cast(analyzer.Analyzer, analyzer.g2h_analyzer)
    rslts, dur, nhash = matcher.match_file(g2h, hash_tab, qry)
    t_hop = 0.02322
    logger.debug(
        f"Matched {qry} ({dur} s, {nhash} hashes) as "
        f"{hash_tab.names[rslts[0][0]]} at {t_hop * float(rslts[0][2])} "
        f"with {rslts[0][1]} of {rslts[0][3]} hashes"
    )


# Run the main function if called from the command line
if __name__ == "__main__":
    localtest()
