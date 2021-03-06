import ujson
import sys
import argparse
import re
import spacy

spacy_nlp = spacy.load('en_core_web_sm')

# tokenize and tag pos
def tokenize_spacy(text):
	tokenized = spacy_nlp(text)
	# use universal pos tags
	toks = [tok.text for tok in tokenized if not tok.is_space]
	pos = [tok.pos_ for tok in tokenized if not tok.is_space]
	lemma = [tok.lemma_.replace(' ','') for tok in tokenized if not tok.is_space]
	lemma = [l if l != '' else t for l, t in zip(lemma, toks)]
	return toks, pos, lemma			


def filter_by_pos(keys, toks, pos, lemma):
	filtered_toks = []
	filtered_pos = []
	filtered_lemma = []
	for t, p, l in zip(toks, pos, lemma):
		if p not in keys:
			filtered_toks.append(t)
			filtered_pos.append(p)
			filtered_lemma.append(l)
	return filtered_toks, filtered_pos, filtered_lemma


def write_to(ls, out_file):
	print('writing to {0}'.format(out_file))
	with open(out_file, 'w+') as f:
		for l in ls:
			f.write((l + '\n'))


def extract(opt, csv_file):
	all_sent1 = []
	all_sent2 = []
	all_sent1_pos = []
	all_sent2_pos = []
	all_sent1_lemma = []
	all_sent2_lemma = []
	all_x_pairs = []	# the pair of words (x1, x2) in sentence template
	max_sent_l = 0

	skip_cnt = 0

	with open(csv_file, 'r') as f:
		line_idx = 0
		for l in f:
			line_idx += 1
			if line_idx == 1 or l.strip() == '':
				continue

			if opt.max_num != -1 and line_idx >= opt.max_num:
				break

			cells = l.rstrip().split(',')
			x1 = cells[2]
			x2 = cells[3]
			sent1 = cells[-2]
			sent2 = cells[-1]

			if opt.tokenize == 1:
				sent1_toks, sent1_pos, sent1_lemma = tokenize_spacy(sent1)
				sent2_toks, sent2_pos, sent2_lemma = tokenize_spacy(sent2)

				if opt.filter != '':
					keys = opt.filter.split(',')
					sent1_toks, sent1_pos, sent1_lemma = filter_by_pos(keys, sent1_toks, sent1_pos, sent1_lemma)
					sent2_toks, sent2_pos, sent1_lemma = filter_by_pos(keys, sent2_toks, sent2_pos, sent2_lemma)

				assert(len(sent1_toks) == len(sent1_pos))
				assert(len(sent2_toks) == len(sent2_pos))
				assert(len(sent1_toks) == len(sent1_lemma))

			else:
				sent1_toks = sent1.split(' ')
				sent2_toks = sent2.split(' ')
			
			max_sent_l = max(max_sent_l, len(sent1_toks), len(sent2_toks))

			all_x_pairs.append('{0} {1}'.format(x1, x2))
			all_sent1.append(' '.join(sent1_toks))
			all_sent2.append(' '.join(sent2_toks))

			all_sent1_pos, all_sent1_lemma, all_sent2_pos, all_sent2_lemma = None, None, None, None
			if opt.tokenize == 1:
				all_sent1_pos.append(' '.join(sent1_pos))
				all_sent2_pos.append(' '.join(sent2_pos))
				all_sent1_lemma.append(' '.join(sent1_lemma))
				all_sent2_lemma.append(' '.join(sent2_lemma))

			if line_idx % 1000 == 0:
				print('extracted {0} examples'.format(line_idx))

	print('skipped {0} examples'.format(skip_cnt))

	return (all_x_pairs, all_sent1, all_sent2, all_sent1_pos, all_sent2_pos, all_sent1_lemma, all_sent2_lemma)


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data', help="Path to unlabeled txt file, generated by generate_template.py", default="data/nli_bias/unlabeled.txt")
parser.add_argument('--output', help="Prefix to the path of output", default="data/nli_bias/unlabeled")
parser.add_argument('--filter', help="List of pos tags to filter out", default="")
parser.add_argument('--max_num', help="The maximum number of examples to extract", type=int, default=-1)
parser.add_argument('--lowercase', help="Whether to use lower case", type=int, default=1)
parser.add_argument('--tokenize', help="Whether to tokenize", type=int, default=0)


def main(args):
	opt = parser.parse_args(args)
	all_x_pairs, all_sent1, all_sent2, all_sent1_pos, all_sent2_pos, all_sent1_lemma, all_sent2_lemma = extract(opt, opt.data)
	print('{0} examples processed.'.format(len(all_sent1)))

	if opt.lowercase == 1:
		all_x_pairs = [l.lower() for l in all_x_pairs]
		all_sent1 = [l.lower() for l in all_sent1]
		all_sent2 = [l.lower() for l in all_sent2]

	write_to(all_x_pairs, opt.output + '.x_pair.txt')
	write_to(all_sent1, opt.output + '.sent1.txt')
	write_to(all_sent2, opt.output + '.sent2.txt')

	if opt.tokenize == 1:
		if opt.lowercase == 1:
			all_sent1_pos = [l.lower() for l in all_sent1_pos]
			all_sent2_pos = [l.lower() for l in all_sent2_pos]
			all_sent1_lemma = [l.lower() for l in all_sent1_lemma]
			all_sent2_lemma = [l.lower() for l in all_sent2_lemma]
		write_to(all_sent1_pos, opt.output + '.sent1_pos.txt')
		write_to(all_sent2_pos, opt.output + '.sent2_pos.txt')
		write_to(all_sent1_lemma, opt.output + '.sent1_lemma.txt')
		write_to(all_sent2_lemma, opt.output + '.sent2_lemma.txt')

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))


