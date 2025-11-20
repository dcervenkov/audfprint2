"""
audfprint.py

Implementation of acoustic-landmark-based robust fingerprinting.
Port of the Matlab implementation.

2014-05-25 Dan Ellis dpwe@ee.columbia.edu
"""


import argparse
import contextlib
import logging
import multiprocessing
import os
import time

import joblib  # type: ignore[import-untyped]

import audfprint_analyze
import audfprint_match

# TODO: My hash_table implementation
import hash_table

time_clock = time.process_time
logger = logging.getLogger("audfprint")


def filename_list_iterator(filelist, wavdir, wavext, listflag):
    """ Iterator to yeild all the filenames, possibly interpreting them
        as list files, prepending wavdir """
    if not listflag:
        for filename in filelist:
            yield os.path.join(wavdir, filename + wavext)
    else:
        for listfilename in filelist:
            with open(listfilename, 'r') as f:
                for filename in f:
                    yield os.path.join(wavdir, filename.rstrip('\n') + wavext)


# for saving precomputed fprints
def ensure_dir(dirname):
    """ ensure that the named directory exists """
    if len(dirname) and not os.path.exists(dirname):
        with contextlib.suppress(OSError):
            os.makedirs(dirname)


# Command line interface

# basic operations, each in a separate function

def file_precompute_peaks_or_hashes(analyzer, filename, precompdir,
                                    precompext=None, hashes_not_peaks=True,
                                    skip_existing=False,
                                    strip_prefix=None):
    """ Perform precompute action for one file, return list
        of message strings """
    # If strip_prefix is specified and matches the start of filename,
    # remove it from filename.
    if strip_prefix and filename[:len(strip_prefix)] == strip_prefix:
        tail_filename = filename[len(strip_prefix):]
    else:
        tail_filename = filename
    # Form the output filename to check if it exists.
    # strip relative directory components from file name
    # Also remove leading absolute path (comp == '')
    relname = '/'.join(
        [
            comp
            for comp in tail_filename.split('/')
            if comp not in ['.', '..', '']
        ]
    )
    root = os.path.splitext(relname)[0]
    if precompext is None:
        if hashes_not_peaks:
            precompext = audfprint_analyze.PRECOMPEXT
        else:
            precompext = audfprint_analyze.PRECOMPPKEXT
    opfname = os.path.join(precompdir, root + precompext)
    if skip_existing and os.path.isfile(opfname):
        return [f"file {opfname} exists (and --skip-existing); skipping"]
    # Do the analysis
    if hashes_not_peaks:
        feature_type = "hashes"
        saver = audfprint_analyze.hashes_save
        output = analyzer.wavfile2hashes(filename)
    else:
        feature_type = "peaks"
        saver = audfprint_analyze.peaks_save
        output = analyzer.wavfile2peaks(filename)
    # save the hashes or peaks file
    if len(output) == 0:
        message = f"Zero length analysis for {filename} -- not saving."
    else:
        # Make sure the output directory exists
        ensure_dir(os.path.split(opfname)[0])
        # Write the file
        saver(opfname, output)
        message = f"wrote {opfname}" + " ( %d %s, %.3f sec)" % (
            len(output),
            feature_type,
            analyzer.soundfiledur,
        )
    return [message]


def file_precompute(analyzer, filename, precompdir, type='peaks', skip_existing=False, strip_prefix=None):
    """ Perform precompute action for one file, return list
        of message strings """
    logger.debug(time.ctime(), "precomputing", type, "for", filename, "...")
    hashes_not_peaks = (type == 'hashes')
    return file_precompute_peaks_or_hashes(analyzer, filename, precompdir,
                                           hashes_not_peaks=hashes_not_peaks,
                                           skip_existing=skip_existing,
                                           strip_prefix=strip_prefix)


def make_ht_from_list(analyzer, filelist, hashbits, depth, maxtime, pipe=None):
    """ Populate a hash table from a list, used as target for
        multiprocess division.  pipe is a pipe over which to push back
        the result, else return it """
    # Create new ht instance
    ht = hash_table.HashTable(hashbits=hashbits, depth=depth, maxtime=maxtime)
    # Add in the files
    for filename in filelist:
        hashes = analyzer.wavfile2hashes(filename)
        ht.store(filename, hashes)
    # Pass back to caller
    if pipe:
        pipe.send(ht)
    else:
        return ht


def do_cmd(cmd, analyzer, hash_tab, filename_iter, matcher, outdir, type, skip_existing=False, strip_prefix=None):
    """ Breaks out the core part of running the command.
        This is just the single-core versions.
    """
    if cmd in ('merge', 'newmerge'):
        # files are other hash tables, merge them in
        for filename in filename_iter:
            hash_tab2 = hash_table.HashTable(filename)
            if "samplerate" in hash_tab.params:
                assert hash_tab.params["samplerate"] == hash_tab2.params["samplerate"]
            else:
                # "newmerge" fails to setup the samplerate param
                hash_tab.params["samplerate"] = hash_tab2.params["samplerate"]
            hash_tab.merge(hash_tab2)

    elif cmd == 'precompute':
        # just precompute fingerprints, single core
        for filename in filename_iter:
            logger.debug(
                file_precompute(
                    analyzer, filename, outdir, type,
                    skip_existing=skip_existing, strip_prefix=strip_prefix
                )
            )

    elif cmd == 'match':
        # Running query, single-core mode
        for num, filename in enumerate(filename_iter):
            msgs = matcher.file_match_to_msgs(analyzer, hash_tab, filename, num)
            for msg in msgs:
                logger.debug(msg)

    elif cmd in ('new', 'add'):
        # Adding files
        tothashes = 0
        for ix, filename in enumerate(filename_iter):
            logger.debug(
                f"{time.ctime()} ingesting #{ix}: {filename} ..."
            )
            dur, nhash = analyzer.ingest(hash_tab, filename)
            tothashes += nhash

        logger.debug(
            f"Added {tothashes} hashes "
            f"({tothashes / float(analyzer.soundfiletotaldur):.1f} hashes/sec)"
        )
    elif cmd == 'remove':
        # Removing files from hash table.
        for filename in filename_iter:
            hash_tab.remove(filename)

    elif cmd == 'list':
        hash_tab.list(lambda x: logger.debug([x]))

    else:
        raise ValueError(f"unrecognized command: {cmd}")


def multiproc_add(analyzer, hash_tab, filename_iter, report, ncores):
    """Run multiple threads adding new files to hash table"""
    # run ncores in parallel to add new files to existing HASH_TABLE
    # lists store per-process parameters
    # Pipes to transfer results
    rx = [[] for _ in range(ncores)]
    tx = [[] for _ in range(ncores)]
    # Process objects
    pr = [[] for _ in range(ncores)]
    # Lists of the distinct files
    filelists = [[] for _ in range(ncores)]
    # unpack all the files into ncores lists
    for ix, filename in enumerate(filename_iter):
        filelists[ix % ncores].append(filename)
    # Launch each of the individual processes
    for ix in range(ncores):
        rx[ix], tx[ix] = multiprocessing.Pipe(False)
        pr[ix] = multiprocessing.Process(target=make_ht_from_list,
                                         args=(analyzer, filelists[ix],
                                               hash_tab.hashbits,
                                               hash_tab.depth,
                                               (1 << hash_tab.maxtimebits),
                                               tx[ix]))
        pr[ix].start()
    # gather results when they all finish
    for core in range(ncores):
        # thread passes back serialized hash table structure
        hash_tabx = rx[core].recv()
        logger.debug(
            f"hash_table {core} has {len(hash_tabx.names)} files "
            f"{sum(hash_tabx.counts)} hashes"
        )
        # merge in all the new items, hash entries
        hash_tab.merge(hash_tabx)
        # finish that thread...
        pr[core].join()


def matcher_file_match_to_msgs(matcher, analyzer, hash_tab, filename):
    """Cover for matcher.file_match_to_msgs so it can be passed to joblib"""
    return matcher.file_match_to_msgs(analyzer, hash_tab, filename)


def do_cmd_multiproc(cmd, analyzer, hash_tab, filename_iter, matcher,
                     outdir, type, report, skip_existing=False,
                     strip_prefix=None, ncores=1):
    """ Run the actual command, using multiple processors """
    if cmd in ('precompute', 'new', 'add'):
        # precompute fingerprints with joblib
        msgslist = joblib.Parallel(n_jobs=ncores)(
                joblib.delayed(file_precompute)(
                    analyzer, file, outdir, type, skip_existing,
                    strip_prefix=strip_prefix
                )
                for file in filename_iter
        )
        # Collapse into a single list of messages
        for msgs in msgslist:
            logger.debug(msgs)

    elif cmd == 'match':
        # Running queries in parallel
        msgslist = joblib.Parallel(n_jobs=ncores)(
                # Would use matcher.file_match_to_msgs(), but you
                # can't use joblib on an instance method
                joblib.delayed(matcher_file_match_to_msgs)(matcher, analyzer,
                                                           hash_tab, filename)
                for filename in filename_iter
        )
        for msgs in msgslist:
            logger.debug(msgs)

    elif cmd in ('new', 'add'):
        # We add by forking multiple parallel threads each running
        # analyzers over different subsets of the file list
        multiproc_add(analyzer, hash_tab, filename_iter, report, ncores)

    else:
        # This is not a multiproc command
        raise ValueError(f"unrecognized multiproc command: {cmd}")


# Command to separate out setting of analyzer parameters
def setup_analyzer(density, is_match, pks_per_frame, fanout, freq_sd, shifts, samplerate, continue_on_error):
    """Create a new analyzer object, taking values from docopts args"""
    # Create analyzer object; parameters will get set below
    analyzer = audfprint_analyze.Analyzer()
    # Read parameters from command line/docopts
    analyzer.density = float(density)
    analyzer.maxpksperframe = int(pks_per_frame)
    analyzer.maxpairsperpeak = int(fanout)
    analyzer.f_sd = float(freq_sd)
    analyzer.shifts = int(shifts)
    # fixed - 512 pt FFT with 256 pt hop at 11025 Hz
    analyzer.target_sr = int(samplerate)
    analyzer.n_fft = 512
    analyzer.n_hop = analyzer.n_fft // 2
    # set default value for shifts depending on mode
    if analyzer.shifts == 0:
        # Default shift is 4 for match, otherwise 1
        analyzer.shifts = 4 if is_match else 1
    analyzer.fail_on_error = not continue_on_error
    return analyzer


def setup_matcher(
    match_win, search_depth, min_count, max_matches, exact_count,
    find_time_range, time_quantile, sortbytime, illustrate, illustrate_hpf
):
    """Create a new matcher objects, set parameters from docopt structure"""
    matcher = audfprint_match.Matcher()
    matcher.window = int(match_win)
    matcher.threshcount = int(min_count)
    matcher.max_returns = int(max_matches)
    matcher.search_depth = int(search_depth)
    matcher.sort_by_time = sortbytime
    matcher.exact_count = exact_count | illustrate | illustrate_hpf
    matcher.illustrate = illustrate | illustrate_hpf
    matcher.illustrate_hpf = illustrate_hpf
    matcher.find_time_range = find_time_range
    matcher.time_quantile = float(time_quantile)
    return matcher


__version__ = 20150406


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Landmark-based audio fingerprinting. Create a new fingerprint dbase with 'new', append "
            "new files to an existing database with 'add', or identify noisy query excerpts with "
            "'match'. 'precompute' writes a *.fpt file under precompdir with precomputed fingerprint "
            "for each input wav file. 'merge' combines previously-created databases into an existing "
            "database; 'newmerge' combines existing databases to create a new one."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )
    parser.add_argument("-?", "--help", action="help", help="Show this help message and exit.")
    parser.add_argument(
        "--version",
        action="version",
        version=f"audfprint-enhanced {__version__}",
        help="Show the program's version number and exit.",
    )
    parser.add_argument(
        "cmd",
        choices=["new", "add", "precompute", "merge", "newmerge", "match", "list", "remove"],
        help="Command to execute.",
    )
    parser.add_argument("-d", "--dbase", help="Fingerprint database file")
    parser.add_argument("-n", "--density", type=float, default=20.0, help="Target hashes per second")
    parser.add_argument("-h", "--hashbits", type=int, default=20, help="How many bits in each hash")
    parser.add_argument("-b", "--bucketsize", type=int, default=100, help="Number of entries per bucket")
    parser.add_argument("-t", "--maxtime", type=int, default=16384, help="Largest time value stored")
    parser.add_argument(
        "-u",
        "--maxtimebits",
        type=int,
        help="maxtime as a number of bits (16384 == 14 bits)",
    )
    parser.add_argument(
        "-r", "--samplerate", type=int, default=11025,
        help="Resample input files to this"
    )
    parser.add_argument(
        "-p", "--precompdir", default=".",
        help="Save precomputed files under this dir"
    )
    parser.add_argument(
        "-i", "--shifts", type=int, default=0,
        help="Use this many subframe shifts building fp"
    )
    parser.add_argument(
        "-w", "--match-win", type=int, default=2,
        help="Maximum tolerable frame skew to count as a match"
    )
    parser.add_argument(
        "-N", "--min-count", type=int, default=5,
        help="Minimum number of matching landmarks to count as a match"
    )
    parser.add_argument(
        "-x", "--max-matches", type=int, default=1,
        help="Maximum number of matches to report for each query"
    )
    parser.add_argument(
        "-X", "--exact-count", action="store_true",
        help="Flag to use more precise (but slower) match counting"
    )
    parser.add_argument(
        "-R", "--find-time-range", action="store_true",
        help="Report the time support of each match"
    )
    parser.add_argument(
        "-Q", "--time-quantile", type=float, default=0.05,
        help="Quantile at extremes of time support"
    )
    parser.add_argument(
        "-S", "--freq-sd", type=float, default=30.0,
        help="Frequency peak spreading SD in bins"
    )
    parser.add_argument(
        "-F", "--fanout", type=int, default=3,
        help="Max number of hash pairs per peak"
    )
    parser.add_argument(
        "-P", "--pks-per-frame", type=int, default=5,
        help="Maximum number of peaks per frame"
    )
    parser.add_argument(
        "-D", "--search-depth", type=int, default=100,
        help="How far down to search raw matching track list"
    )
    parser.add_argument(
        "-H", "--ncores", type=int, default=1,
        help="Number of processes to use"
    )
    parser.add_argument(
        "-o", "--opfile", default="",
        help="Write output (matches) to this file, not stdout"
    )
    parser.add_argument(
        "-K", "--precompute-peaks", action="store_true",
        help="Precompute just landmarks (else full hashes)"
    )
    parser.add_argument(
        "-k", "--skip-existing", action="store_true",
        help="On precompute, skip items if output file already exists"
    )
    parser.add_argument(
        "-C", "--continue-on-error", action="store_true",
        help="Keep processing despite errors reading input"
    )
    parser.add_argument(
        "-l", "--list", dest="list", action="store_true",
        help="Input files are lists, not audio"
    )
    parser.add_argument(
        "-T", "--sortbytime", action="store_true",
        help="Sort multiple hits per file by time (instead of score)"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Set logging level to debug (default is warning)",
    )
    parser.add_argument(
        "-I", "--illustrate", action="store_true",
        help="Make a plot showing the match"
    )
    parser.add_argument(
        "-J", "--illustrate-hpf", action="store_true",
        help="Plot the match, using onset enhancement"
    )
    parser.add_argument(
        "-W", "--wavdir", default="",
        help="Find sound files under this dir"
    )
    parser.add_argument(
        "-V", "--wavext", default="",
        help="Extension to add to wav file names"
    )
    parser.add_argument("file", nargs="*", help="Files or lists (with --list) to process")
    return parser


def parse_args(argv=None):
    """Create and parse the command-line arguments."""
    parser = build_parser()
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.getLogger().setLevel(logging.DEBUG if args.verbose else logging.WARNING)

    initticks = time_clock()

    maxtimebits = args.maxtimebits if args.maxtimebits is not None else hash_table._bitsfor(args.maxtime)

    # Setup the analyzer if we're using one (i.e., unless "merge")
    analyzer = setup_analyzer(
        args.density,
        args.cmd == "match",
        args.pks_per_frame,
        args.fanout,
        args.freq_sd,
        args.shifts,
        args.samplerate,
        args.continue_on_error,
    ) if args.cmd not in ["merge", "newmerge", "list", "remove"] else None

    precomp_type = 'peaks' if args.precompute_peaks else 'hashes'

    # Set up the hash table, if we're using one (i.e., unless "precompute")
    if args.cmd != "precompute":
        # For everything other than precompute, we need a database name
        # Check we have one
        if not args.dbase:
            raise ValueError("dbase name must be provided if not precompute")
        if args.cmd in ["new", "newmerge"]:
            # Check that the output directory can be created before we start
            ensure_dir(os.path.split(args.dbase)[0])
            # Create a new hash table
            hash_tab = hash_table.HashTable(
                    hashbits=args.hashbits,
                    depth=args.bucketsize,
                    maxtime=(1 << maxtimebits))
            # Set its samplerate param
            if analyzer:
                hash_tab.params['samplerate'] = analyzer.target_sr

        else:
            logger.debug(f"{time.ctime()} Reading hash table {args.dbase}")
            hash_tab = hash_table.HashTable(args.dbase)
            if analyzer and 'samplerate' in hash_tab.params \
                    and hash_tab.params['samplerate'] != analyzer.target_sr:
                logger.debug("db samplerate overridden to ", analyzer.target_sr)
    else:
        # The command IS precompute
        # dummy empty hash table
        hash_tab = None

    # Create a matcher
    matcher = setup_matcher(
        args.match_win,
        args.search_depth,
        args.min_count,
        args.max_matches,
        args.exact_count,
        args.find_time_range,
        args.time_quantile,
        args.sortbytime,
        args.illustrate,
        args.illustrate_hpf,
    ) if args.cmd == 'match' else None

    filename_iter = filename_list_iterator(
            args.file, args.wavdir, args.wavext, args.list)

    #######################
    # Run the main commmand
    #######################

    # How many processors to use (multiprocessing)
    if args.ncores > 1 and args.cmd not in ["merge", "newmerge", "list", "remove"]:
        # merge/newmerge/list/remove are always single-thread processes
        do_cmd_multiproc(args.cmd, analyzer, hash_tab, filename_iter,
                         matcher, args.precompdir,
                         precomp_type,
                         skip_existing=args.skip_existing,
                         strip_prefix=args.wavdir,
                         ncores=args.ncores)
    else:
        do_cmd(args.cmd, analyzer, hash_tab, filename_iter,
               matcher, args.precompdir, precomp_type,
               skip_existing=args.skip_existing,
               strip_prefix=args.wavdir)

    elapsedtime = time_clock() - initticks
    if analyzer and analyzer.soundfiletotaldur > 0.:
        logger.debug("Processed "
              + "%d files (%.1f s total dur) in %.1f s sec = %.3f x RT" \
              % (analyzer.soundfilecount, analyzer.soundfiletotaldur,
                 elapsedtime, (elapsedtime / analyzer.soundfiletotaldur)))

    # Save the hash table file if it has been modified
    if hash_tab and hash_tab.dirty:
        # We already created the directory, if "new".
        hash_tab.save(args.dbase)


if __name__ == '__main__':
    main()
