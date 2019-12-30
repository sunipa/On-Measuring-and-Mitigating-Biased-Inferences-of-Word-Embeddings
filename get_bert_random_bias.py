import sys
import os
import argparse
import numpy as np
import h5py
import itertools
from collections import defaultdict
import json
import torch
from random import gauss

def gen_gaussian_bias(dim):
	vec = [gauss(0, 1) for i in range(dim)]
	mag = sum(x**2 for x in vec) ** .5
	return np.asarray([x/mag for x in vec], dtype=np.float32)


def process(opt):
	for i in range(opt.num):
		comp = gen_gaussian_bias(opt.dim)

		f = h5py.File(opt.output + 'rand{0}.hdf5'.format(i), "w")		
		f["bias"] = comp
		f.close()

	
def main(arguments):
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--dim', help="The dim", type=int, default = 768)
	parser.add_argument('--num', help="The number of random biases to generate", type=int, default = 4)
	parser.add_argument('--output', help="Prefix of the output file names. ", type=str, default = "")
	opt = parser.parse_args(arguments)

	process(opt)

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))
