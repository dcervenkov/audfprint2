# coding=utf-8
import cProfile
import pstats

from audfprint import cli

argv = ["match", "-d", "tmp.fpdb.hdf", "data/query.mp3"]

cProfile.run('cli.main(argv)', 'fpmstats')

p = pstats.Stats('fpmstats')

p.sort_stats('time')
p.print_stats(10)
