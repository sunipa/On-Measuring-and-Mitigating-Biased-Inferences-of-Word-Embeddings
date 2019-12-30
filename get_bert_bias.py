import sys
sys.path.insert(0, '../pytorch-pretrained-BERT')
import os
import argparse
import numpy as np
import h5py
import itertools
from collections import defaultdict
import json
import torch
from torch import cuda
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

def component(word1, word2):
	word1 =  word1/np.linalg.norm(word1); word2 = word2/np.linalg.norm(word2)
	w = (word1 - word2)
	w = w/np.linalg.norm(w)
	return w

def get_bert_vec(opt):
	print('loading BERT tokenizer...')
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

	words = [opt.word1, opt.word2]
	toks = []
	for w in words:
		toks.append(tokenizer.tokenize(w)[0])	# only take the first token/wordpiece
	tok_idx = tokenizer.convert_tokens_to_ids(toks)

	print('loading BERT model...')
	bert = BertModel.from_pretrained('bert-base-uncased')
	print('verifying BERT model...')
	bert.eval()

	tok_idx = torch.from_numpy(np.asarray(tok_idx))
	emb = bert.embeddings.word_embeddings(tok_idx)
	emb = emb.data.cpu()
	assert(emb.shape == (2, 768))

	a = emb[0].view(-1)
	b = emb[1].view(-1)
	dot = torch.dot(a, b)
	a_norm = torch.sqrt(torch.dot(a, a))
	b_norm = torch.sqrt(torch.dot(b, b))
	cos_sim = dot / a_norm / b_norm
	print('wv1: {0}'.format(a))
	print('wv2: {0}'.format(b))
	print('cos sim: {0}'.format(cos_sim))

	comp = component(emb[0].numpy(), emb[1].numpy())
	return comp


def process(opt):
	comp = get_bert_vec(opt)

	f = h5py.File(opt.output, "w")		
	f["bias"] = comp
	f.close()

	
def main(arguments):
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--word1', help="word1", default = "he")
	parser.add_argument('--word2', help="word2", default = "she")
	parser.add_argument('--output', help="Prefix of the output file names. ", type=str, default = "")
	opt = parser.parse_args(arguments)

	process(opt)

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))
