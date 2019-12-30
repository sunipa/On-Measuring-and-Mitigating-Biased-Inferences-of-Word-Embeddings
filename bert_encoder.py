import sys
sys.path.insert(0, '../pytorch-pretrained-BERT')
import torch
from torch import cuda
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from torch import nn
from torch.autograd import Variable
from holder import *
from util import *

# encoder with Elmo
class BertEncoder(torch.nn.Module):
	def __init__(self, opt, shared):
		super(BertEncoder, self).__init__()
		self.opt = opt
		self.shared = shared

		# if dynamic debiasing
		#	hard wired here, TODO
		if self.opt.dynamic_debias == 1:
			print('loading BERT tokenizer...')
			tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
			words = ['he', 'she']
			toks = []
			for w in words:
				toks.append(tokenizer.tokenize(w)[0])	# only take the first token/wordpiece
			debias_keyword_idx = tokenizer.convert_tokens_to_ids(toks)
			debias_keyword_idx = torch.from_numpy(np.asarray(debias_keyword_idx))
			self.debias_keyword_idx = to_device(debias_keyword_idx, self.opt.bert_gpuid)

		if self.opt.debias == 1 and self.opt.dynamic_debias == 0:
			print('loading bias vector from {0}...'.format(self.opt.bias))
			bias_f = h5py.File(self.opt.bias, 'r')
			self.bias = torch.from_numpy(bias_f['bias'][:]).float()
			self.bias = to_device(self.bias, self.opt.bert_gpuid)
			if self.opt.fp16 == 1:
				self.bias = self.bias.half()
		
		print('loading BERT model...')
		self.bert = BertModel.from_pretrained('bert-base-uncased')

		print('verifying BERT model...')
		self.bert.eval()

		for n in self.bert.children():
			for p in n.parameters():
				p.skip_init = True
				p.is_bert = True	# tag as bert fields

		# if to lock bert
		if opt.fix_bert == 1:
			for n in self.bert.children():
				for p in n.parameters():
					p.requires_grad = False

		self.customize_cuda_id = self.opt.bert_gpuid
		self.fp16 = opt.fp16 == 1

		self.zero = Variable(torch.zeros(1), requires_grad=False)
		self.zero = to_device(self.zero, self.opt.bert_gpuid)
		if self.opt.fp16 == 1:
			self.zero = self.zero.half()


	def get_seg_mask(self):
		mask = torch.ones(self.shared.batch_l, self.shared.sent_l1+self.shared.sent_l2-1).long()
		mask = to_device(mask, self.opt.bert_gpuid)

		seg1 = torch.zeros(self.shared.batch_l, self.shared.sent_l1).long()
		seg2 = torch.ones(self.shared.batch_l, self.shared.sent_l2-1).long()	# removing the heading [CLS]
		seg = torch.cat([seg1, seg2], 1)
		seg = to_device(seg, self.opt.bert_gpuid)

		return seg, mask


	def _bert_debiased(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
		# input bias has shape (batch_l, 1, bert_size)
		def _bert_embeddings_debiased(bias, input_ids, token_type_ids=None):
			seq_length = input_ids.size(1)
			position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
			position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
			if token_type_ids is None:
			    token_type_ids = torch.zeros_like(input_ids)

			words_embeddings = self.bert.embeddings.word_embeddings(input_ids)
			position_embeddings = self.bert.embeddings.position_embeddings(position_ids)
			token_type_embeddings = self.bert.embeddings.token_type_embeddings(token_type_ids)

			# debiasing only the word embeddings
			batch_l, seq_l = input_ids.shape
			prod = torch.bmm(words_embeddings, bias.transpose(1,2))	# (batch_l, seq_l, 1)
			words_embeddings = words_embeddings - prod * bias

			embeddings = words_embeddings + position_embeddings + token_type_embeddings
			embeddings = self.bert.embeddings.LayerNorm(embeddings)
			embeddings = self.bert.embeddings.dropout(embeddings)
			return embeddings

		if attention_mask is None:
			attention_mask = torch.ones_like(input_ids)
		if token_type_ids is None:
			token_type_ids = torch.zeros_like(input_ids)

		# We create a 3D attention mask from a 2D tensor mask.
		# Sizes are [batch_size, 1, 1, to_seq_length]
		# So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
		# this attention mask is more simple than the triangular masking of causal attention
		# used in OpenAI GPT, we just need to prepare the broadcast dimension here.
		extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

		# Since attention_mask is 1.0 for positions we want to attend and 0.0 for
		# masked positions, this operation will create a tensor which is 0.0 for
		# positions we want to attend and -10000.0 for masked positions.
		# Since we are adding it to the raw scores before the softmax, this is
		# effectively the same as removing these entirely.
		extended_attention_mask = extended_attention_mask.to(dtype=next(self.bert.parameters()).dtype) # fp16 compatibility
		extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
		
		# replace the original with debiased version
		#embedding_output = self.bert.embeddings(input_ids, token_type_ids)
		if self.opt.dynamic_debias == 0:
			bias = self.bias.view(1, 1, self.opt.bert_size).expand(self.shared.batch_l, 1, self.opt.bert_size)
			embedding_output = _bert_embeddings_debiased(bias, input_ids, token_type_ids)
		else:
			kw_emb = self.bert.embeddings.word_embeddings(self.debias_keyword_idx)
			bias = component(kw_emb[0], kw_emb[1])
			bias = bias.view(1, 1, self.opt.bert_size).expand(self.shared.batch_l, 1, self.opt.bert_size)
			embedding_output = _bert_embeddings_debiased(bias, input_ids, token_type_ids)
		# continue as usual

		encoded_layers = self.bert.encoder(embedding_output,
			extended_attention_mask,
			output_all_encoded_layers=output_all_encoded_layers)
		sequence_output = encoded_layers[-1]
		pooled_output = self.bert.pooler(sequence_output)
		if not output_all_encoded_layers:
			encoded_layers = encoded_layers[-1]
		return encoded_layers, pooled_output


	def forward(self, sent1, sent2, char_sent1, char_sent2, bert1, bert2):
		bert1 = to_device(bert1, self.opt.bert_gpuid)
		bert2 = to_device(bert2, self.opt.bert_gpuid)
		bert_tok = torch.cat([bert1, bert2[:, 1:]], 1)	# removing the heading [CLS]

		seg, mask = self.get_seg_mask()

		assert(seg.shape[1] == bert_tok.shape[1])

		if self.opt.fix_bert == 1:
			with torch.no_grad():
				if self.opt.debias == 0:
					last, pooled = self.bert(bert_tok, seg, mask, output_all_encoded_layers=False)
				else:
					last, pooled = self._bert_debiased(bert_tok, seg, mask, output_all_encoded_layers=False)
		else:
			if self.opt.debias == 0:
				last, pooled = self.bert(bert_tok, seg, mask, output_all_encoded_layers=False)
			else:
				last, pooled = self._bert_debiased(bert_tok, seg, mask, output_all_encoded_layers=False)

		last = last + pooled.unsqueeze(1) * self.zero

		# move to the original device
		last = to_device(last, self.opt.gpuid)

		self.shared.bert_enc = last
		
		return last


	def begin_pass(self):
		pass

	def end_pass(self):
		pass


