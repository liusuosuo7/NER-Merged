# -*- coding: utf-8 -*
import codecs
import os
import pickle
from .evaluate_metric import evaluate_ByCategory, get_chunks_onesent, evaluate_chunk_level


class DataReader():
	def __init__(self, dataname, file_dir, classes, fmodels, fn_stand_res):
		self.dataname = dataname
		self.file_dir = file_dir
		self.fmodels = fmodels
		self.classes = classes
		self.fn_stand_res = fn_stand_res

	def read_seqModel_data(self, fn, column_no=-1, delimiter=' '):
		# read seq model's results
		word_sequences = list()
		tag_sequences = list()
		total_word_sequences = list()
		total_tag_sequences = list()
		with codecs.open(fn, 'r', 'utf-8') as f:
			lines = f.readlines()
		curr_words = list()
		curr_tags = list()
		for k in range(len(lines)):
			line = lines[k].strip()
			if len(line) == 0 or line.startswith('-DOCSTART-'): # new sentence or new document
				if len(curr_words) > 0:
					word_sequences.append(curr_words)
					tag_sequences.append(curr_tags)
					curr_words = list()
					curr_tags = list()
				continue

			strings = line.split(delimiter)
			word = strings[0].strip()
			tag = strings[column_no].strip()  # be default, we take the last tag
			if tag=='work' or tag=='creative-work': # for wnut17
				tag='work'
			if self.dataname=='ptb2':
				tag='B-'+tag
			curr_words.append(word)
			curr_tags.append(tag)
			total_word_sequences.append(word)
			total_tag_sequences.append(tag)
			if k == len(lines) - 1:
				word_sequences.append(curr_words)
				tag_sequences.append(curr_tags)

		return word_sequences, tag_sequences

	def read_model_pred(self, fn_pred):
		# read spanNER model's results
		with open(fn_pred, 'rb') as f:
			pred_chunks, true_chunks = pickle.load(f)
		return pred_chunks, true_chunks

	def read_linkner_results(self, fn_results):
		"""读取LinkNER的结果文件"""
		try:
			with open(fn_results, 'rb') as f:
				data = pickle.load(f)
			
			# LinkNER结果可能有不同的格式，这里进行适配
			if isinstance(data, dict):
				pred_chunks = data.get('pred_chunks', [])
				true_chunks = data.get('true_chunks', [])
				uncertainties = data.get('uncertainties', [])
				return pred_chunks, true_chunks, uncertainties
			elif isinstance(data, (list, tuple)) and len(data) >= 2:
				return data[0], data[1], data[2] if len(data) > 2 else []
			else:
				raise ValueError("Unexpected data format in LinkNER results")
		except Exception as e:
			print(f"Error reading LinkNER results from {fn_results}: {e}")
			return [], [], []

	def get_allModels_pred(self):
		tchunks_models = []
		pchunks_models = []
		for fmodel in self.fmodels:
			fn_model_res = os.path.join(self.file_dir, fmodel)
			
			# 根据文件扩展名判断读取方式
			if fn_model_res.endswith('.pkl'):
				if 'linkner' in fmodel.lower():
					# LinkNER结果格式
					pred_chunks, true_chunks, _ = self.read_linkner_results(fn_model_res)
				else:
					# 标准spanNER结果格式
					pred_chunks, true_chunks = self.read_model_pred(fn_model_res)
			else:
				# 序列标注格式
				word_sequences, tag_sequences = self.read_seqModel_data(fn_model_res)
				pred_chunks, true_chunks = self.convert_sequences_to_chunks(word_sequences, tag_sequences)
			
			pchunks_models.append(pred_chunks)
			tchunks_models.append(true_chunks)

		# 获取唯一的真实chunks
		tchunks_unique = []
		for tchunks in tchunks_models:
			tchunks_unique.extend(tchunks)
		tchunks_unique = list(set(tchunks_unique))

		# 处理预测chunks
		tchunks_models_onedim = []
		pchunks_models_onedim = []
		pchunk2label_models = []
		tchunk2label_dic = {}

		for i, (pchunks, tchunks) in enumerate(zip(pchunks_models, tchunks_models)):
			pchunks_models_onedim.extend(pchunks)
			tchunks_models_onedim.extend(tchunks)
			
			# 构建chunk到label的映射
			pchunk2label = {}
			for chunk in pchunks:
				if len(chunk) >= 4:
					label, sid, eid, sentid = chunk[:4]
					key = (sid, eid, sentid)
					pchunk2label[key] = label
			pchunk2label_models.append(pchunk2label)
			
			# 构建真实chunk到label的映射
			for chunk in tchunks:
				if len(chunk) >= 4:
					label, sid, eid, sentid = chunk[:4]
					key = (sid, eid, sentid)
					tchunk2label_dic[key] = label

		# 计算每个模型在每个类别上的F1分数
		class2f1_models = []
		for pchunks, tchunks in zip(pchunks_models, tchunks_models):
			class2f1 = evaluate_ByCategory(pchunks, tchunks, self.classes)
			class2f1_models.append(class2f1)

		return (tchunks_models, tchunks_unique, pchunks_models,
				tchunks_models_onedim, pchunks_models_onedim,
				pchunk2label_models, tchunk2label_dic, class2f1_models)

	def convert_sequences_to_chunks(self, word_sequences, tag_sequences):
		"""将序列标注格式转换为chunk格式"""
		pred_chunks = []
		true_chunks = []
		
		for sent_id, (words, tags) in enumerate(zip(word_sequences, tag_sequences)):
			chunks = get_chunks_onesent(tags)
			for chunk in chunks:
				label, start, end = chunk
				pred_chunks.append((label, start, end, sent_id))
				true_chunks.append((label, start, end, sent_id))
		
		return pred_chunks, true_chunks

	def get_model_f1s(self):
		"""获取每个模型的F1分数"""
		f1s = []
		for fmodel in self.fmodels:
			fn_model_res = os.path.join(self.file_dir, fmodel)
			
			if fn_model_res.endswith('.pkl'):
				if 'linkner' in fmodel.lower():
					pred_chunks, true_chunks, _ = self.read_linkner_results(fn_model_res)
				else:
					pred_chunks, true_chunks = self.read_model_pred(fn_model_res)
			else:
				word_sequences, tag_sequences = self.read_seqModel_data(fn_model_res)
				pred_chunks, true_chunks = self.convert_sequences_to_chunks(word_sequences, tag_sequences)
			
			f1, _, _, _, _, _ = evaluate_chunk_level(pred_chunks, true_chunks)
			f1s.append(f1)
		
		return f1s