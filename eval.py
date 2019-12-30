import sys
sys.path.insert(0, '../pytorch-pretrained-BERT')
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
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dir', help="Path to the data dir", default="data/bert_nli/")
parser.add_argument('--data', help="Path to training data hdf5 file.", default="snli.test.hdf5")
parser.add_argument('--word_vecs', help="The path to word embeddings", default = "glove.hdf5")
parser.add_argument('--char_idx', help="The path to word2char index file", default = "char.idx.hdf5")
parser.add_argument('--dict', help="The path to word dictionary", default = "snli.word.dict")
parser.add_argument('--char_dict', help="The path to char dictionary", default = "char.dict.txt")
parser.add_argument('--load_file', help="Path to where model to be loaded.", default="")
# resource specs
parser.add_argument('--res', help="Path to validation resource files, seperated by comma.", default="")
## pipeline specs
parser.add_argument('--debias', help="Whether to debias embeddings", type=int, default=0)
parser.add_argument('--dynamic_debias', help="Whether to do dynamic debiasing", type=int, default=0)
parser.add_argument('--bias', help="Path to the bias vector hdf5", default="")
parser.add_argument('--gpuid', help="The GPU index, if -1 then use CPU", type=int, default=-1)
parser.add_argument('--use_word_vec', help="Whether to use word vec", type=int, default=0)
parser.add_argument('--use_char_emb', help="Whether to use character embedding", type=int, default=0)
parser.add_argument('--word_vec_size', help="The input word embedding dim", type=int, default=300)
parser.add_argument('--hidden_size', help="The general hidden size of the pipeline", type=int, default=768)
parser.add_argument('--num_char', help="The number of char", type=int, default=68)
parser.add_argument('--token_l', help="The maximal token length", type=int, default=16)
parser.add_argument('--dropout', help="The dropout probability", type=float, default=0.0)
parser.add_argument('--num_label', help="The number of label", type=int, default=3)
# bert specs
parser.add_argument('--bert_gpuid', help="The GPU index for bert, if -1 then use CPU", type=int, default=-1)
parser.add_argument('--fp16', help="Whether to use fp16 format", type=int, default=1)
parser.add_argument('--fix_bert', help="Whether to fix bert update", type=int, default=1)
parser.add_argument('--bert_size', help="The input elmo dim", type=int, default=768)
parser.add_argument('--use_cached_bert', help="Whether to use cached bert embeddings", type=int, default=0)
## pipeline stages
parser.add_argument('--enc', help="The type of encoder, bert", default='bert')
parser.add_argument('--cls', help="The type of classifier, linear", default='linear')
parser.add_argument('--loss', help="The type of loss, boundary", default='boundary')
#
parser.add_argument('--verbose', help="Whether to print out every prediction", type=int, default=0)

# 
def forward(opt, m, source, target, char_source, char_target, bert1, bert2, batch_ex_idx, batch_l, source_l, target_l, res_map):
	wv_idx1 = Variable(source, requires_grad=False)
	wv_idx2 = Variable(target, requires_grad=False)
	cv_idx1 = Variable(char_source, requires_grad=False)
	cv_idx2 = Variable(char_target, requires_grad=False)
	# update network parameters
	m.update_context(batch_ex_idx, batch_l, source_l, target_l, res_map)

		# forward pass
	with torch.no_grad():
		pred = m.forward(wv_idx1, wv_idx2, cv_idx1, cv_idx2, bert1, bert2)

	return pred

def evaluate(opt, shared, m, data):
	m.train(False)

	val_loss = 0.0
	num_ex = 0

	loss = MulticlassLoss(opt, shared)

	val_idx, val_num_ex = data.subsample(1.0)
	data_size = val_idx.size()[0]
	print('evaluating on {0} batches {1} examples'.format(data_size, val_num_ex))

	loss.begin_pass()
	m.begin_pass()
	for i in range(data_size):

		(data_name, source, target, char_source, char_target, bert1, bert2,
			batch_ex_idx, batch_l, source_l, target_l, label, res_map) = data[i]
		pred = forward(opt, m, source, target, char_source, char_target, bert1, bert2,
			batch_ex_idx, batch_l, source_l, target_l, res_map)

		y_gold = Variable(label, requires_grad=False)

		# loss
		batch_loss = loss(pred, y_gold)

		# stats
		val_loss += float(batch_loss.data)
		num_ex += batch_l

		if (i+1) % 2000 == 0:
			print('evaluated {0} batches'.format(i+1))

	perf, extra_perf = loss.get_epoch_metric()
	m.end_pass()
	loss.end_pass()
	print('finished evaluation on {0} examples'.format(num_ex))

	return (perf, extra_perf, val_loss / num_ex, num_ex)

# TODO, HARD CODED
def bias_analysis(opt, shared, m):
	print('running bias analysis...')
	print('loading BERT tokenizer...')
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
	words = ['he', 'she']
	toks = []
	for w in words:
		toks.append(tokenizer.tokenize(w)[0])	# only take the first token/wordpiece
	debias_keyword_idx = tokenizer.convert_tokens_to_ids(toks)
	debias_keyword_idx = torch.from_numpy(np.asarray(debias_keyword_idx))
	debias_keyword_idx = to_device(debias_keyword_idx, opt.bert_gpuid)
	kw_emb = m.encoder.bert.embeddings.word_embeddings(debias_keyword_idx)
	a = kw_emb[0].view(-1)
	b = kw_emb[1].view(-1)
	cos_sim = cosine_sim(a, b)
	print('he: {0}'.format(a))
	print('she: {0}'.format(b))
	print('cos_sim btw he and she: {0}'.format(cos_sim))

	baseline_bias = './data/bert_bias/gender_bert.hdf5'
	bias_f = h5py.File(baseline_bias, 'r')
	baseline_bias = torch.from_numpy(bias_f['bias'][:]).float()
	baseline_bias = to_device(baseline_bias, opt.bert_gpuid)
	cur_bias = component(a, b).float()
	cos_sim = cosine_sim(cur_bias, baseline_bias)
	print('cos_sim btw cur_bias and baseline_bias: {0}'.format(cos_sim))






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


	if opt.gpuid != -1:
		torch.cuda.manual_seed_all(1)

	# build model
	m = Pipeline(opt, shared)

	# initialization
	print('loading pretrained model from {0}...'.format(opt.load_file))
	param_dict = load_param_dict('{0}.hdf5'.format(opt.load_file))
	m.set_param_dict(param_dict)

	if opt.gpuid != -1:
		m.distribute()	# distribute to multigpu

	# loading data
	res_files = None if opt.res == '' else opt.res.split(',')
	data = Data(opt, opt.data, res_files)

	#
	perf, extra_perf, avg_loss, num_ex = evaluate(opt, shared, m, data)
	extra_perf_str = ' '.join(['{:.4f}'.format(p) for p in extra_perf])
	print('Val {0:.4f} Extra {1} Loss: {2:.4f}'.format(
		perf, extra_perf_str, avg_loss))

	bias_analysis(opt, shared, m)

	#print('saving model to {0}'.format('tmp'))
	#param_dict = m.get_param_dict()
	#save_param_dict(param_dict, '{0}.hdf5'.format('tmp'))


if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))