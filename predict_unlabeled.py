import sys
from pipeline import *
import argparse
import h5py
import os
import random
import time
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from torch import cuda
from holder import *
from data import *
from multiclass_loss import *

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dir', help="Path to the data dir", default="data/nli_bias/")
parser.add_argument('--data', help="Path to validation data hdf5 file.", default="unlabeled.hdf5")
parser.add_argument('--res', help="Path to training resource files, seperated by comma.", default="")
parser.add_argument('--word_vecs', help="The path to word embeddings", default = "unlabeled_glove.hdf5")
parser.add_argument('--char_idx', help="The path to word2char index file", default = "unlabeled_char.idx.hdf5")
parser.add_argument('--dict', help="The path to word dictionary", default = "unlabeled.word.dict")
parser.add_argument('--char_dict', help="The path to char dictionary", default = "unlabeled_char.dict.txt")
parser.add_argument('--load_file', help="Path to where model to be loaded.", default="")
# for bias
parser.add_argument('--debias', help="Whether to debias embeddings", type=int, default=0)
parser.add_argument('--dynamic_debias', help="Whether to do dynamic debiasing", type=int, default=0)
parser.add_argument('--bias', help="Path to the bias vector hdf5", default="")
# generic parameter
parser.add_argument('--param_init_type', help="The type of parameter initialization", default='xavier_uniform')
parser.add_argument('--param_init', help="The scale of the normal distribution from which weights are initialized", type=float, default=0.01)
parser.add_argument('--use_char_emb', help="Whether to use char emb", type=int, default=0)
parser.add_argument('--use_word_vec', help="Whether to use word vec", type=int, default=0)
parser.add_argument('--fix_word_vecs', help="Whether to make word embeddings NOT learnable", type=int, default=1)
parser.add_argument('--dropout', help="The dropout probability", type=float, default=0.0)
parser.add_argument('--seed', help="The random seed", type=int, default=3435)
parser.add_argument('--num_char', help="The number of distinct chars", type=int, default=68)
parser.add_argument('--hidden_size', help="The general hidden size of the pipeline", type=int, default=768)
parser.add_argument('--word_vec_size', help="The input word embedding dim", type=int, default=300)
parser.add_argument('--token_l', help="The maximal token length", type=int, default=16)
## pipeline specs
parser.add_argument('--gpuid', help="The GPU index, if -1 then use CPU", type=int, default=-1)
parser.add_argument('--bert_gpuid', help="The GPU index for bert, if -1 then use CPU", type=int, default=-1)
parser.add_argument('--fp16', help="Whether to use fp16 format", type=int, default=1)
parser.add_argument('--fix_bert', help="Whether to fix bert update", type=int, default=1)
parser.add_argument('--bert_size', help="The input bert dim", type=int, default=768)
parser.add_argument('--enc', help="The type of encoder, bert", default='bert')
parser.add_argument('--cls', help="The type of classifier, linear", default='linear')
parser.add_argument('--num_label', help="The number of prediction labels", type=int, default=3)
# specs for unlabeled
parser.add_argument('--pred_output', help="Prediction output file", default='./models/unlabeled_pred.txt')

def write_log(path, lines):
	print('writing log to {0}'.format(path))
	with open(path, 'w') as f:
		for l in lines:
			f.write(l + '\n')


def evaluate(opt, shared, m, data):
	m.train(False)

	batch_cnt = 0
	val_loss = 0.0
	num_ex = 0

	loss = MulticlassLoss(opt, shared)

	val_idx, val_num_ex = data.subsample(1.0)
	data_size = val_idx.size()[0]
	print('evaluating on {0} batches {1} examples'.format(data_size, val_num_ex))

	log = ['x1,x2,premise,hypothesis,entail_probability,neutral_probability,contradiction_probability']

	m.begin_pass()
	for i in range(data_size):
		(data_name, source, target, char_source, char_target, bert1, bert2,
			batch_ex_idx, batch_l, source_l, target_l, label, res_map) = data[i]

		wv_idx1 = Variable(source, requires_grad=False)
		wv_idx2 = Variable(target, requires_grad=False)
		cv_idx1 = Variable(char_source, requires_grad=False)
		cv_idx2 = Variable(char_target, requires_grad=False)
		y_gold = Variable(label, requires_grad=False)

		# update network parameters
		m.update_context(batch_ex_idx, batch_l, source_l, target_l, res_map)

		# forward pass
		pred = m.forward(wv_idx1, wv_idx2, cv_idx1, cv_idx2, bert1, bert2)

		# loss
		batch_loss = loss(pred, y_gold)

		# stats
		val_loss += float(batch_loss.data)
		num_ex += batch_l

		# logging
		dist = pred.data.exp()
		for k, ex_idx in enumerate(batch_ex_idx):
			# output format is: premise, hypothesis, p(E), p(N), p(C)
			log.append('{0},{1},{2},{3},{4:.4f},{5:.4f},{6:.4f}'.format(res_map['x_pair'][k][0], res_map['x_pair'][k][1], ' '.join(res_map['sent1'][k]), ' '.join(res_map['sent2'][k]), float(dist[k][0]), float(dist[k][1]), float(dist[k][2])))

		if (batch_cnt + 1) % 1000 == 0:
			print('predicted {0} batches'.format(batch_cnt + 1))
		batch_cnt += 1

	perf, extra_perf = loss.get_epoch_metric()
	m.end_pass()
	print('finished evaluation on {0} examples'.format(num_ex))

	# printing
	write_log(opt.pred_output, log)

	return (perf, extra_perf, val_loss / num_ex, num_ex)



def main(args):
	opt = parser.parse_args(args)
	shared = Holder()

	# 
	opt.data = opt.dir + opt.data
	opt.res = '' if opt.res == ''  else ','.join([opt.dir + path for path in opt.res.split(',')])
	opt.word_vecs = opt.dir + opt.word_vecs
	opt.char_idx = opt.dir + opt.char_idx
	opt.dict = opt.dir + opt.dict
	opt.char_dict = opt.dir + opt.char_dict
	opt.bias = opt.dir + opt.bias

	torch.manual_seed(opt.seed)
	if opt.gpuid != -1:
		torch.cuda.set_device(opt.gpuid)
		torch.cuda.manual_seed_all(opt.seed)

	print(opt)

	# build model
	m = Pipeline(opt, shared)

	# initializing from pretrained
	if opt.load_file != '':
		m.init_weight()
		print('loading pretrained model from {0}...'.format(opt.load_file))
		param_dict = load_param_dict('{0}.hdf5'.format(opt.load_file))
		m.set_param_dict(param_dict)
	else:
		m.init_weight()
		model_parameters = filter(lambda p: p.requires_grad, m.parameters())
		num_params = sum([np.prod(p.size()) for p in model_parameters])
		print('total number of trainable parameters: {0}'.format(num_params))
	
	if opt.gpuid != -1:
		m.distribute()	# distribute to multigpu

	# loading data
	res_files = None if opt.res == '' else opt.res.split(',')
	data = Data(opt, opt.data, res_files)

	#
	perf, extra_perf, avg_loss, num_ex = evaluate(opt, shared, m, data)
	extra_perf_str = ' '.join(['{:.4f}'.format(p) for p in extra_perf])


if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))
