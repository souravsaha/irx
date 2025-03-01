import csv
import sys

sys.path.append('models')
sys.path.append('Datasets')
sys.path.append('utilities')
import json
import math
import random
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import ir_explain.explainers.listwise.simple_explainers
import numpy as np
import torch
from ir_explain.explainers.listwise.simple_explainers import (get_explainer,
                                                              multi_rank)
from ir_explain.utils import utility
from ir_explain.utils.optimization import kendalltau_concord
from pyserini.analysis import Analyzer, get_lucene_analyzer
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)
project_dir = Path.cwd()
seed = 100

analyzer = utility.load_analyzer()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



class Explain(object):
    def __init__(self, hparams: Dict[str, Any]): 

        """ Init the base explainer object, load the model to be explained and the data object.
            Args: hparams: a dictionary of hyperparameters and necessary objects.
                index_dir: the directory of the corpus index, pre-built using pyserini.
                RankModel: the model to be explained.
                InferenceDataset: the dataset object for inference, without y label.
                dataIterate: the dataIterate object for inference data, for easier doc perturbation and reranking.
                queries: a dictionary of queries, with q_id as key and query as value.
        """
        print('Initiating indexes...')  
        self.index_reader = utility.loader_index(hparams['index_dir'])
        # this is now dense ranked list 
        # either a list of doc ids or qid -> [[docs]]
        self.model = hparams['RankModel']
        
        #print('self.model ', self.model)
        #self.InferenceDataset = hparams['InferenceDataset']
        
        #print('self.InferenceDataset ', self.InferenceDataset)
        #self.dataIterate = hparams['dataIterate']
        
        #print('self.dataIterate ', self.dataIterate)
        self.query = hparams['query']
        #print('self.queries ', self.queries)
        
    def _init_query(self, q_id: str, rank_scores: bool= False):
        """Earlier: load the query, tokens, get the prediction scores of each doc in the list"""

        """load the query, tokens, get the prediction scores of each doc in the list"""
        self.InferenceDataset.__init_q_docs__(q_id, self.queries[q_id])
        self.InferenceDataset.query_tokens = [q for q in analyzer.analyze(self.InferenceDataset.query) if q not in utility.STOP]
        print(f'query: {self.queries[q_id]}')
        print(f'top_docs : {len(self.InferenceDataset.top_docs)}')
        
        #if rank_scores:
            # get the prediction scores of each doc in the list
        #    prediction = self._rank_docs(self.InferenceDataset.query, self.InferenceDataset.top_docs)
        #    self.InferenceDataset.prediction = prediction
        #    self.InferenceDataset.rank = np.argsort(-self.InferenceDataset.prediction) # sort doc index from high to low

    def _rank_docs(self, query:str, docs: List[str], batch_size = 64):
        """prediction of the documents"""
        inputs_data = self.dataIterate.CustomDataset(query, docs, self.InferenceDataset.tokenizer, device)
        inputs_iter = DataLoader(inputs_data, batch_size = batch_size, collate_fn=getattr(inputs_data, 'collate_fn', None))
        prediction = np.array([])
        with torch.no_grad():
            for i, batch in enumerate(inputs_iter):
                #print('batch : ', batch)
                #print(type(batch))
                #print(len(batch))
                #for item in batch:
                out = self.model(batch).detach().cpu().squeeze(-1).numpy()
                #out = self.model(item)
                #print(out.last_hidden_state.shape)
                #op = torch.nn.functional.softmax(out.logits, dim = -1)
                #print(op)get_explainer
                #exit(1)
                #print(out.shape)
                prediction = np.append(prediction, out)
        return prediction

    def refine_candidates_by_perturb(self, replaced_tokens: Dict[str, float], doc_id: str, doc:str) -> Dict[str, float]:
        """ Compute the candidate tokens which influence the document the most by masking the token in the document and comparing prediction diff.
            Mask a token and create a doc D, for all token create docs, compute their scores
            based on this score rank the masked token
        """
        replaced_tokens = list(replaced_tokens.keys())
        input_orig = self.InferenceDataset.__buildFromDoc__(doc)
        score_orig = self.model(input_orig).detach().cpu().item()
        new_docs = []
        for replace_token in replaced_tokens:
            new_docs.append(doc.replace(f' {replace_token} ', '[UNK]'))   # keep blank before and after the token to avoid characters.
        
        prediction = self._rank_docs(self.InferenceDataset.query, new_docs)
        score_diff = abs(score_orig - prediction)
    
        refined = dict((k, v) for k, v in zip(replaced_tokens, score_diff))
        refined = sorted(refined.items(), key=lambda kv: kv[1], reverse=True)
        return dict(refined)

    def refine_candidates_by_bm25(self, replaced_tokens: Dict[str, float], doc_id: str, doc: str) -> Dict[str, float]:
        """
        Refine the candidates based on their bm25 score for a particular document D
        """
        replaced_tokens = list(replaced_tokens.keys())
        print('replaced_tokens : ', replaced_tokens)
        # for each token compute their bm25 score for that particular document doc_id
        #analyzer = Analyzer(get_lucene_analyzer(stemmer='porter'))
        bm25_scores = [self.index_reader.compute_bm25_term_weight(doc_id, q, analyzer=get_lucene_analyzer(stemmer='porter')) for q in replaced_tokens]
        #bm25_scores = [self.index_reader.compute_bm25_term_weight(doc_id, q, analyzer=None) for q in replaced_tokens]
        term_score = dict((k, v) for k, v in zip(replaced_tokens, bm25_scores))
        term_score = sorted(term_score.items(), key=lambda kv: kv[1], reverse=True)
        print('term_score:  ', term_score)
        return dict(term_score)

    def get_candidates_reranker(self, q_id: str, topd: int, topk: int, topr: int, dense_ranking, method: str='bm25') -> Dict[str, float]:
        """
        Generate `top_r` many candidates 
        """
        if method == 'bm25':
            refine_method = self.refine_candidates_by_bm25
        elif method == 'perturb':
            pass
            #refine_method = self.refine_candidates_by_perturb
        elif method == 'None':
            refine_method = lambda x, y, z: x   # only return the first argument.
        else:
            raise ValueError('Invalid candidates selecting method.')
        candidates_scores = {}
        print('refine_method: ', refine_method)
        #self._init_query(q_id)

        # expects dense_ranking to be a list of docids
        top_docs_id = dense_ranking[:topd]
        top_docs = []
        for docid in top_docs_id:
            #doc = json.loads(self.index_reader.doc_raw(docid))['contents']
            analyzer = Analyzer(get_lucene_analyzer(stemmer='porter'))
            tokens = analyzer.analyze(json.loads(self.index_reader.doc_raw(docid))['contents'])
            doc= ' '.join(map(str, tokens))
            #doc = json.loads(self.index_reader.doc_raw(docid))['contents']
            top_docs.append(doc)


        #print('self.InferenceDataset.top_docs_id[:topd] : ', self.InferenceDataset.top_docs_id[:topd])
        #print('self.InferenceDataset.top_docs[:topd] : ', self.InferenceDataset.top_docs[:topd])
        
        #for doc_id, doc in tqdm(zip(self.InferenceDataset.top_docs_id[:topd], self.InferenceDataset.top_docs[:topd]), desc='Perturb each doc...'):
        for doc_id, doc in tqdm(zip(top_docs_id[:topd], top_docs[:topd]), desc='Perturb each doc...'):
            print('doc_id : ', doc_id)
            print('topk : ', topk)
            candidates_tfidf = utility.get_candidates(self.index_reader, doc_id, topk)
            candidates_refined = refine_method(candidates_tfidf, doc_id, doc)
            for k, v in candidates_refined.items():
                if k in candidates_scores:
                    candidates_scores[k] += [v]
                else:
                    candidates_scores[k] = [v]
        # average scores
        for k, v in candidates_scores.items():
            candidates_scores[k] = sum(v)/len(v)

        candidates_scores = sorted(candidates_scores.items(), key=lambda kv: kv[1], reverse=True)
        refined_candidates = dict(candidates_scores[:topr])
        
        print('refined_candidates : ', refined_candidates)
        
        return refined_candidates

    def sample_doc_pair(self, dense_ranking, dense_ranking_score, ranked: int=20, m: int=500, style: str='random', tolerance: float=2.0) -> List[Tuple[int, int]]:
        """
        Sample document pairs d_i, d_j, additionally taking dense ranking score as input
        """
        length = len(dense_ranking)
        if style == 'random':       # generate mC2 pairs
            #pairs = list(combinations(range(self.InferenceDataset.length), 2))
            pairs = list(combinations(range(length), 2))
            print('initial list: ', len(pairs))
        elif style == 'topk_random':
            #assert(ranked <= self.InferenceDataset.length)
            assert(ranked <= length)
            ranked_list = list(range(ranked))
            #tail_list = list(range(ranked, self.InferenceDataset.length))
            tail_list = list(range(ranked, length))
            pairs = [(a, b) for a in ranked_list for b in tail_list]
        else:
            raise ValueError(f'Not supported style {style}')
        # filter our pairs with prediction scores diffence < tolerance, e.g., 0.01
        # TODO 2:
        #rank = np.argsort(-self.InferenceDataset.prediction)
        # TODO : reverse list 
        rank = np.argsort(dense_ranking_score)

        #probs_diff = np.array([self.InferenceDataset.prediction[rank[h]] - self.InferenceDataset.prediction[rank[l]] for h, l in pairs])
        # TODO : reverse here also
        probs_diff = np.array([float(dense_ranking_score[rank[l]]) - float(dense_ranking_score[rank[h]]) for h, l in pairs])
        valid_index = list(np.where(probs_diff >= tolerance)[0])
        #valid_index = list(np.where(probs_diff >= 0.01)[0])
        print(f'Difference : {probs_diff}')
        pairs = [pairs[i] for i in valid_index]
        random.seed(seed)
        if len(pairs) < m:
            m = len(pairs)
        pairs = random.sample(pairs, m)
        print('final list: ', len(pairs))
        print(pairs)
        return pairs

    def build_matrix(self, dense_ranking, candidates: List[str], pairs: List[Tuple[int]], EXP_model: str='language_model') -> List[List[float]]:
        """
        1. Get the explainer for which you want to compute score
        2. 
        """
        # need to find candidates cooccur in both docs.
        explainer = get_explainer(EXP_model)
        matrix = []
        print(f'Sampled {len(pairs)} doc pairs')

        for rank_h_id, rank_l_id in tqdm(pairs, desc="building matrix for doc pairs..."):
            weight = 1 + math.log(rank_l_id - rank_h_id)    
            # TODO 3
            #doc_h_id = self.InferenceDataset.rank[rank_h_id]
            doc_h_id = dense_ranking[rank_h_id]
            
            #doc_l_id = self.InferenceDataset.rank[rank_l_id]
            doc_l_id = dense_ranking[rank_l_id]
            
            #doc_h = self.InferenceDataset.top_docs[doc_h_id]
            doc_h = json.loads(self.index_reader.doc_raw(doc_h_id))['contents']
            
            #doc_l = self.InferenceDataset.top_docs[doc_l_id]
            doc_l = json.loads(self.index_reader.doc_raw(doc_l_id))['contents']
            #doc_l = dense_ranking_score[doc_l_id]
            
            s_h = explainer(candidates, doc_h, analyzer)
            s_l = explainer(candidates, doc_l, analyzer )


            concordance = (np.array(s_h) - np.array(s_l)) * np.array(weight)
            matrix.append(concordance.tolist())
        # reshape matrix to candidate dimension first. 
        matrix = np.array(matrix).transpose(1, 0).tolist()
        return matrix

    def evaluate_fidelity(self, dense_ranking, dense_ranking_score, expansions: List[str], EXP_model: List[str], top_k: int=10, vote: int=2, tolerance: float=2.0) -> Tuple[float, float, float, float]:
        """
        Kendall tau evaluation
        what does it return?
        """
        print('Kendalltau evaluation...')
        # TODO 
        #prediction_orig = self.InferenceDataset.prediction.copy()     
        #rank = self.InferenceDataset.rank.copy()

        prediction_orig = np.array(dense_ranking_score, dtype=float)  
        rank_temp = np.argsort(dense_ranking_score)
        rank = rank_temp[::-1]                       # reverse the list

        # expects dense_ranking to be a list of docids
        top_docs_id = dense_ranking[:top_k]
        print(f'topdocs fomred... {top_docs_id}')

        top_docs = []
        for docid in top_docs_id:
            doc = json.loads(self.index_reader.doc_raw(docid))['contents']
            top_docs.append(doc)

        print(f'exptraction of topdocs completed, size of topdocs {len(top_docs_id)}')

        #pred_orig_topk = prediction_orig[rank[:top_k]]
        pred_orig_topk = prediction_orig[:top_k]
        print(f'prediction orig topk {pred_orig_topk}')
        if isinstance(EXP_model, str):
            # single-ranker
            EXP_model = [EXP_model]   
        # prediction_new = explainers.multi_rank(EXP_model, expansions, top_docs, analyzer)
        prediction_new = multi_rank(EXP_model, expansions, top_docs, analyzer)
        print(f'new prediction {prediction_new}')
        print(f'size of new prediction {len(prediction_new)}, {type(prediction_new)}')

        if len(EXP_model) <= 1:     # consider just single ranker
            #pred_new_topk = prediction_new[rank[:top_k]]
            pred_new_topk = prediction_new[:top_k]
            prediction_orig = prediction_orig[:top_k] 
            print(f'computing kendall tau..')
            correl_g = kendalltau_concord.kendalltau(prediction_orig, prediction_new).correlation
            print(f'completed kendall tau 1 ..')
            correl_l = kendalltau_concord.kendalltau(pred_orig_topk, pred_new_topk).correlation
            print(f'completed kendall tau 2 ..')
            correl_tg = kendalltau_concord.kendalltau_gap(prediction_orig, prediction_new, tolerance)
            print(f'completed kendall tau 3 ..')
            correl_tl = kendalltau_concord.kendalltau_gap(pred_orig_topk, pred_new_topk, tolerance)
            print(f'completed kendall tau 4 ..')
        else:
            # multi vote, consider all explainers.
            #pred_new_topk_all = [pred[rank[:top_k]] for pred in prediction_new]
            pred_new_topk_all = [pred[:top_k] for pred in prediction_new]
            
            prediction_orig = prediction_orig[:top_k]

            correl_g = kendalltau_concord.coverage_multi(prediction_orig, prediction_new, vote=vote, tolerance=0)
            correl_l = kendalltau_concord.coverage_multi(pred_orig_topk, pred_new_topk_all, vote=vote, tolerance=0)
            correl_tg = kendalltau_concord.coverage_multi(prediction_orig, prediction_new, vote=vote, tolerance=tolerance)
            correl_tl = kendalltau_concord.coverage_multi(pred_orig_topk, pred_new_topk_all, vote=vote, tolerance=tolerance)
        return correl_g, correl_l, correl_tg, correl_tl 

    
    def explain(hparams: Dict[str, Any]) -> List[str]:
        """
        A pipeline including candidates, matrix generation, and extract intent using geno solver
        Args:

        Return :
            a list of terms as explanation
        """
