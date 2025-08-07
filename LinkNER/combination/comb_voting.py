# -*- coding: utf-8 -*

import numpy as np
import os
import pickle
from .dataread import DataReader
import json


def evaluate_chunk_level(pred_chunks,true_chunks):
	correct_preds, total_correct, total_preds = 0., 0., 0.
	correct_preds += len(set(true_chunks) & set(pred_chunks))
	total_preds += len(pred_chunks)
	total_correct += len(true_chunks)
	p = correct_preds / total_preds if correct_preds > 0 else 0
	r = correct_preds / total_correct if correct_preds > 0 else 0
	f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0

	return f1, p, r,correct_preds, total_preds, total_correct


class CombByVoting():
	def __init__(self, dataname, file_dir, fmodels, f1s, cmodelname, classes, fn_stand_res, fn_prob, result_dir="combination/comb_result"):
		self.dataname = dataname
		self.file_dir = file_dir
		self.fmodels = fmodels
		self.f1s = f1s
		self.cmodelname = cmodelname
		self.fn_prob = fn_prob
		self.classes = classes
		self.result_dir = result_dir

		# 确保结果目录存在
		os.makedirs(self.result_dir, exist_ok=True)

		self.mres = DataReader(dataname, file_dir, classes, fmodels, fn_stand_res)

		self.wf1 = 1.0
		self.wscore = 0.8

	def get_unique_pchunk_labs(self):
		tchunks_models,\
		tchunks_unique, \
		pchunks_models, \
		tchunks_models_onedim, \
		pchunks_models_onedim, \
		pchunk2label_models, \
		tchunk2label_dic, \
		class2f1_models=self.mres.get_allModels_pred()
		self.tchunks_unique = tchunks_unique
		self.class2f1_models = class2f1_models
		self.tchunk2label_dic = tchunk2label_dic

		# the unique chunk that predict by the model..
		pchunks_unique = list(set(pchunks_models_onedim))

		# get the unique non-O chunk's label that are predicted by all the models.
		keep_pref_upchunks = []
		pchunk_plb_ms = []
		for pchunk in pchunks_unique:
			lab, sid, eid, sentid = pchunk
			key1 = (sid, eid, sentid)
			if key1 not in keep_pref_upchunks:
				keep_pref_upchunks.append(key1)
				plb_ms = [] # the length is the num of the models
				# the first position is the pchunk
				for i in range(len(self.f1s)):
					plb = 'O'
					if key1 in pchunk2label_models[i]:
						plb = pchunk2label_models[i][key1]
					plb_ms.append(plb)
				pchunk_plb_ms.append(plb_ms)

		# get the non-O true chunk that are not be recognized..
		for tchunk in tchunks_unique:
			if tchunk not in pchunks_unique: # it means that the tchunk are not been recognized by all the models
				plab, sid, eid, sentid = tchunk
				key1 = (sid, eid, sentid)
				if key1 not in keep_pref_upchunks:
					continue

		return pchunk_plb_ms, keep_pref_upchunks

	def best_potential(self):
		pchunk_plb_ms, keep_pref_upchunks = self.get_unique_pchunk_labs()
		assert len(pchunk_plb_ms) == len(keep_pref_upchunks)

		comb_kchunks = []
		for pchunk_plb_m, pref_upchunks in zip(pchunk_plb_ms, keep_pref_upchunks):
			sid, eid, sentid = pref_upchunks
			key1 = (sid, eid, sentid)
			if key1 in self.tchunk2label_dic:
				klb = self.tchunk2label_dic[key1]
			elif 'O' in pchunk_plb_m:
				klb = 'O'
			else:
				klb = pchunk_plb_m[0]
			if klb != 'O':
				kchunk = (klb, sid, eid, sentid)
				comb_kchunks.append(kchunk)

		f1, p, r, correct_preds, total_preds, total_correct = evaluate_chunk_level(comb_kchunks, self.tchunks_unique)
		print()
		print('best_potential results: ')
		print("f1, p, r, correct_preds, total_preds, total_correct:")
		print(f1, p, r, correct_preds, total_preds, total_correct)

		return [f1, p, r, correct_preds, total_preds, total_correct]

	def voting_majority(self):
		pchunk_plb_ms, keep_pref_upchunks = self.get_unique_pchunk_labs()
		assert len(pchunk_plb_ms) == len(keep_pref_upchunks)

		comb_kchunks = []
		for pchunk_plb_m, pref_upchunks in zip(pchunk_plb_ms, keep_pref_upchunks):
			lb2num_dic = {}
			for plbm in pchunk_plb_m:
				if plbm not in lb2num_dic:
					lb2num_dic[plbm] = 0.0
				lb2num_dic[plbm] += 1

			klb = sorted(lb2num_dic, key=lambda x: lb2num_dic[x])[-1]
			if klb != 'O':
				sid, eid, sentid = pref_upchunks
				kchunk = (klb, sid, eid, sentid)
				comb_kchunks.append(kchunk)
		comb_kchunks = list(set(comb_kchunks))
		f1, p, r, correct_preds, total_preds, total_correct = evaluate_chunk_level(comb_kchunks, self.tchunks_unique)
		print()
		print('majority_voting results: ')
		print("f1, p, r, correct_preds, total_preds, total_correct:")
		print(f1, p, r, correct_preds, total_preds, total_correct)

		kf1 = int(f1 * 10000)
		fn_save_comb_kchunks = os.path.join(self.result_dir, f'VM_combine_{kf1}.pkl')

		pickle.dump([comb_kchunks, self.tchunks_unique], open(fn_save_comb_kchunks, "wb"))

		return [f1, p, r, correct_preds, total_preds, total_correct]

	def voting_weightByOverallF1(self):
		pchunk_plb_ms, keep_pref_upchunks = self.get_unique_pchunk_labs()
		assert len(pchunk_plb_ms) == len(keep_pref_upchunks)

		comb_kchunks = []
		for pchunk_plb_m, pref_upchunks in zip(pchunk_plb_ms, keep_pref_upchunks):
			lb2num_dic = {}
			for plbm, f1 in zip(pchunk_plb_m, self.f1s):
				if plbm not in lb2num_dic:
					lb2num_dic[plbm] = 0.0
				lb2num_dic[plbm] += f1
			klb = sorted(lb2num_dic, key=lambda x: lb2num_dic[x])[-1]
			if klb != 'O':
				sid, eid, sentid = pref_upchunks
				kchunk = (klb, sid, eid, sentid)
				comb_kchunks.append(kchunk)

		f1, p, r, correct_preds, total_preds, total_correct = evaluate_chunk_level(comb_kchunks, self.tchunks_unique)
		print()
		print('voting_weightByOverallF1 results: ')
		print("f1, p, r, correct_preds, total_preds, total_correct:")
		print(f1, p, r, correct_preds, total_preds, total_correct)

		kf1 = int(f1 * 10000)
		fn_save_comb_kchunks = os.path.join(self.result_dir, f'VOF1_combine_{kf1}.pkl')
		pickle.dump([comb_kchunks, self.tchunks_unique], open(fn_save_comb_kchunks, "wb"))

		return [f1, p, r, correct_preds, total_preds, total_correct]

	def voting_weightByCategotyF1(self):
		pchunk_plb_ms, keep_pref_upchunks = self.get_unique_pchunk_labs()
		assert len(pchunk_plb_ms) == len(keep_pref_upchunks)

		comb_kchunks = []
		for pchunk_plb_m, pref_upchunks in zip(pchunk_plb_ms, keep_pref_upchunks):
			lb2num_dic = {}
			for plbm, f1, cf1_dic in zip(pchunk_plb_m, self.f1s, self.class2f1_models):
				if plbm not in lb2num_dic:
					lb2num_dic[plbm] = 0.0
				if plbm == 'O':
					lb2num_dic[plbm] += f1
				else:
					lb2num_dic[plbm] += cf1_dic[plbm]
			klb = sorted(lb2num_dic, key=lambda x: lb2num_dic[x])[-1]
			if klb != 'O':
				sid, eid, sentid = pref_upchunks
				kchunk = (klb, sid, eid, sentid)
				comb_kchunks.append(kchunk)

		f1, p, r, correct_preds, total_preds, total_correct = evaluate_chunk_level(comb_kchunks, self.tchunks_unique)
		print()
		print('voting_weightByCategotyF1 results: ')
		print("f1, p, r, correct_preds, total_preds, total_correct:")
		print(f1, p, r, correct_preds, total_preds, total_correct)

		kf1 = int(f1 * 10000)
		fn_save_comb_kchunks = os.path.join(self.result_dir, f'VCF1_combine_{kf1}.pkl')
		pickle.dump([comb_kchunks, self.tchunks_unique], open(fn_save_comb_kchunks, "wb"))

		return [f1, p, r, correct_preds, total_preds, total_correct]

	def voting_prob(self):
		"""基于概率的投票方法，适用于LinkNER的不确定性估计"""
		pchunk_plb_ms, keep_pref_upchunks = self.get_unique_pchunk_labs()
		assert len(pchunk_plb_ms) == len(keep_pref_upchunks)

		# 加载概率信息
		prob_data = {}
		if self.fn_prob and os.path.exists(self.fn_prob):
			with open(self.fn_prob, 'rb') as f:
				prob_data = pickle.load(f)

		comb_kchunks = []
		for pchunk_plb_m, pref_upchunks in zip(pchunk_plb_ms, keep_pref_upchunks):
			lb2prob_dic = {}
			for i, plbm in enumerate(pchunk_plb_m):
				if plbm not in lb2prob_dic:
					lb2prob_dic[plbm] = 0.0
				
				# 如果有概率信息，使用概率加权
				prob_weight = 1.0
				if pref_upchunks in prob_data and i < len(prob_data[pref_upchunks]):
					prob_weight = prob_data[pref_upchunks][i].get(plbm, 0.0)
				
				lb2prob_dic[plbm] += prob_weight * self.f1s[i]

			klb = sorted(lb2prob_dic, key=lambda x: lb2prob_dic[x])[-1]
			if klb != 'O':
				sid, eid, sentid = pref_upchunks
				kchunk = (klb, sid, eid, sentid)
				comb_kchunks.append(kchunk)

		f1, p, r, correct_preds, total_preds, total_correct = evaluate_chunk_level(comb_kchunks, self.tchunks_unique)
		print()
		print('voting_prob results: ')
		print("f1, p, r, correct_preds, total_preds, total_correct:")
		print(f1, p, r, correct_preds, total_preds, total_correct)

		kf1 = int(f1 * 10000)
		fn_save_comb_kchunks = os.path.join(self.result_dir, f'VPROB_combine_{kf1}.pkl')
		pickle.dump([comb_kchunks, self.tchunks_unique], open(fn_save_comb_kchunks, "wb"))

		return [f1, p, r, correct_preds, total_preds, total_correct]

	def combine_results(self, method='voting_majority'):
		"""统一的结果组合接口"""
		method_map = {
			'best_potential': self.best_potential,
			'voting_majority': self.voting_majority,
			'voting_weightByOverallF1': self.voting_weightByOverallF1,
			'voting_weightByCategotyF1': self.voting_weightByCategotyF1,
			'voting_prob': self.voting_prob
		}
		
		if method in method_map:
			return method_map[method]()
		else:
			raise ValueError(f"Unknown combination method: {method}")