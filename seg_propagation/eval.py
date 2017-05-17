#!/usr/bin/env python

# ----------------------------------------------------------------------------
# A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation
#-----------------------------------------------------------------------------
# Copyright (c) 2016 Federico Perazzi
# Licensed under the BSD License
# Written by Federico Perazzi
# ----------------------------------------------------------------------------

"""
Evaluate a technique and store results in HDF5 file.

"""

import os
import time
import argparse

import numpy   as np
import os.path as osp

import sys
sys.path.insert(0, '../lib/davis/python/lib/')


from prettytable import PrettyTable
from davis.dataset import db_eval,db_save_eval
from davis import cfg,log

def parse_args():
	"""Parse input arguments."""

	parser = argparse.ArgumentParser(
			description="""Evaluate a technique and store results.
			""")

	parser.add_argument(
			dest='input',default=None,type=str,
			help='Path to the technique to be evaluated')

	parser.add_argument(
			dest='output',default=None,type=str,
			help='Output folder')

	parser.add_argument(
			'--metrics',default=None,nargs='+',type=str,choices=['J','F','T'])

	args = parser.parse_args()

	return args

if __name__ == '__main__':

	args       = parse_args()
	args.input = osp.abspath(args.input)

	log.info("Evaluation sequences: %s" % os.listdir(args.input))

	import pdb
	pdb.set_trace()


	db_eval_dict = db_eval(osp.basename(args.input),
			os.listdir(args.input),osp.dirname(args.input),args.metrics)


	import pdb
	pdb.set_trace()


	log.info("Saving results in: %s"%osp.join(
			args.output,osp.basename(args.input))+".h5")

	db_save_eval(db_eval_dict,outputdir=args.output)
