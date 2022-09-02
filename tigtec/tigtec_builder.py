import matplotlib.pyplot as plt
import math 
import os
import pandas as pd
import numpy as np
import datetime
import time
from copy import deepcopy
from tqdm.notebook import tqdm
import string  
import seaborn as sns
import tqdm


#PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

#Sentence transformers
from sentence_transformers import SentenceTransformer
from scipy import spatial

#T5 grammar correction
import happytransformer


#graph library
import networkx as nx

#nlp library fom BLEU score
import nltk


#colour function on torch_text_classifier
from  tigtec.torch_text_classifier import plot_change

class tigtec:
    def __init__(self,
                classifier,
                mlm,
                n : int,
                attribution: str = 'attention',
                explo_strategy: str = 'static',
                sentence_similarity = None,
                topk: int = 15,
                mask_variety: int = 3,
                margin: float = 0.2,
                beam_width = 3,
                alpha = 0.5):
        
        self.classifier = classifier
        self.mlm = mlm
        self.n = n
        self.attribution = attribution
        self.explo_strategy = explo_strategy
        self.sentence_similarity = sentence_similarity
        if sentence_similarity == 'sentence_transformer' :
            self.sentence_transformer = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.topk = topk
        self.mask_variety = mask_variety
        self.margin = margin
        self.beam_width = beam_width
        self.alpha = alpha
        
        #List of cf & cf informations
        self.graph_cf = []
        self.cf_list = []
        self.cf_html_list = []
        self.reviews = []
        
    def mlm_inference(self, masked_text) :
        inputs = self.classifier.tokenizer(masked_text, return_tensors='pt')
  
        with torch.no_grad():
            logits = self.mlm(**inputs).logits

        # retrieve index of [MASK]
        mask_token_index = (inputs.input_ids == self.classifier.tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

        #   predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        logit  = logits[0, mask_token_index]
        predicted_tokens_id = torch.topk(logit.flatten(), self.topk).indices
        words = self.classifier.tokenizer.decode(predicted_tokens_id)
        print(words)
        words = words.split(" ")
        # punct_remove_list = ["#", ",", ";", "!", "?", "'", ".", "."]
        
        # for word in words :
        #     if (word[-1] in punct_remove_list) | (word[0] in punct_remove_list) :
        #         words.remove(word)
        #     else : 
        #         pass
    
        return(words)
    
    def sentence_transformer_similarity(self, text1, text2) :
        sentences = text1 + text2
        embeddings = self.sentence_transformer.encode(sentences)
        similarity = 1 - spatial.distance.cosine(embeddings[0], embeddings[1])
        
        return(similarity)
    
    def cf_cost(self, init_review, cf_review, target) :
        init_pred = self.classifier.predict(init_review)
        cf_pred = self.classifier.predict(cf_review)
        idx_init_state = np.argmax(init_pred)
        target_cf_state = target
        # target_cf_state = np.argmin(init_pred)
        cf_state = np.argmax(cf_pred)
        #   score_dist = np.abs(init_pred-cf_pred)[:,0]
        #   score_dist = (init_pred-cf_pred)[:,idx_init_state]
        score_target = cf_pred[0][target_cf_state]
        similarity = 0
        
        if self.sentence_similarity == "cls_embedding" :  
            similarity += self.classifier.cls_similarity(init_review, cf_review)
            
        if self.sentence_similarity == "sentence_transformer" :
            similarity += self.sentence_transformer_similarity(init_review, cf_review)
        
        else :
            pass
            
        #   cost = score_dist + similarity
        cost = - (score_target + similarity*(self.alpha))
        
        return(cost, similarity)

    def replace_token(self, review, to_mask, target):
      

        #   #Reconstitution de la review
        old_review = ' '.join(review)
        #Récupération du token devant êtr masqué
        #   token_max = token_attribution.iloc[to_mask,0]
        print(review)
        print(to_mask)
        token_max = review[to_mask]
        new_review = review.copy()
        new_review[to_mask] = self.classifier.tokenizer.mask_token


        
        #Constitution de la nouvelle review avec le mask, et MLM inférence pour remplacement du mask par le nouveau token
        new_review = ' '.join(new_review)
        new_tokens = self.mlm_inference(masked_text=[new_review])
        
        #Suppression du token en train d'être remplacé de la liste de tokens inférée par MLM
        if token_max in new_tokens :
            new_tokens.remove(token_max)
        
        dist_list = []
        for t in new_tokens :
            #Remplacer iter review par init review pour maximiser la distance par rapport au point de départ
            cost  = self.cf_cost([old_review], [new_review.replace("[MASK]", t)], target)[0]
            dist_list.append(cost)
        
        #   review_max = np.argmax(dist_list)
        #   cost_max = max(dist_list)
        #   review_min = np.argmin(dist_list)
        #   cost_min = min(dist_list)
        cost_dict = {i : j for i,j in enumerate (dist_list)}
        cost_dict = {i: j for i, j in sorted(cost_dict.items(), key=lambda item: item[1])}
        
        costs_min = list(cost_dict.values())[0:self.mask_variety]
        reviews_min = list(cost_dict.keys())[0:self.mask_variety]
        
        new_reviews_tokenized = []
        new_reviews = []
        
        for review_min in reviews_min :
            new_review_tokenized = review.copy()
            new_review_tokenized[to_mask] = new_tokens[review_min]
            new_review = ' '.join(new_review_tokenized)
            
            new_reviews_tokenized.append(new_review_tokenized)
            new_reviews.append(new_review)
        
        new_tokens_variety = [new_tokens[t] for t in reviews_min]

        
        return(new_reviews, new_reviews_tokenized, token_max, new_tokens_variety)
    
    def generate_cf(self, review, target):
      #Prédictions text initial
        init_pred = self.classifier.predict(review)
        nb_class = init_pred.shape[1]
        init_state = np.argmax(init_pred)
        target_state = target
        # target_state = np.argmin(init_pred)
        init_cost = init_pred[0][target_state]
        if self.sentence_similarity is not None :
            init_cost += 1
        
        #Initialisation de l'état de prédiction du CF généré
        cf_state = init_state.copy()
        cf_review = review.copy()
        cf_pred = init_pred.copy()
        #   i=0
        #   token_list_encoded = [t for t in tokenizer.encode(review[0]) if t not in [101, 102, 103]]
        
        #Initialisation du graph basé sur le text initial
        attribution_coeff = self.classifier.compute_token_importance(text=cf_review)
        text_initial_tokenized = attribution_coeff['token'].tolist()
        #   text_initial_tokenized = [tokenizer.decode(t).replace(" ", "") for t in token_list_encoded]
        G_text = nx.DiGraph()
        G_text.add_node(0, text = text_initial_tokenized, hist_mask = [], hist_mask_text = [], attrib_coeff = 1, cost = init_cost, similarity = 1, state=init_state, cf = False)
        wait_list = [(0,1)]
        indx=0
        
        #Premier test : on itère jusqu'à la profondeur max en monde beamsearch
        #   for depth in range(len(text_initial_tokenized)) :
        nb_cf = 0  
        while nb_cf < self.n :
            i = wait_list[0][0]
            #Récupération historique des tokens masqués et du text du noeud parent
            predecessor_hist_mask = G_text.nodes.data()[i]['hist_mask']
            predecessor_text_masked = G_text.nodes.data()[i]['hist_mask_text']
            predecessor_text = G_text.nodes.data()[i]['text']
            
            if (len(text_initial_tokenized) == len(predecessor_hist_mask)) :
                break
            
            if self.explo_strategy == 'evolutive' :
                attribution_coeff = self.classifier.compute_token_importance(text=[' '.join(predecessor_text)])
            
            #On filtre l'attribution en enlevant les tokens déjà masqués/remplacés
            attribution_iter = attribution_coeff[attribution_coeff.index.isin(predecessor_hist_mask)==False]
            #to do : penser au cas de figure avec des attribution égales
            ind_to_mask = attribution_iter[attribution_iter['token'].isin([".", ",", ";"])==False].nlargest(self.beam_width, 'Attribution coefficient')['token'].index.tolist()
            print(predecessor_hist_mask)
            print(wait_list)
            #Pour chaque token à changer, chacun dans un nouveau noeud
            for j in ind_to_mask :
                if (nb_cf == self.n) :
                    break
        #       indx+=1
                predecessor_hist_mask_iter = predecessor_hist_mask.copy()
                predecessor_hist_mask_iter.append(j)
                predecessor_text_masked_iter = predecessor_text_masked.copy()
                text_iter = predecessor_text.copy()

                new_reviews, new_reviews_tokenized, old_token, new_tokens = self.replace_token(text_iter,j, target_state)

                #Ajout à l'historique des tokens masqués
                predecessor_text_masked_iter.append(old_token)
                for k in range(len(new_reviews)) :
                    if (nb_cf == self.n) :
                        break
                    indx+=1
                    #Nouvelle prédiction
                    cf_pred_iter = self.classifier.predict([new_reviews[k]])
                    cf_state_iter = np.argmax(cf_pred_iter)
                    #on garde le cf si la pred est > à la moyenne + une marge, et que c'est bien la pred la plus élevée
                    cf_to_keep_iter = (cf_pred_iter[0][target_state] >= 1/nb_class  + self.margin) & (cf_pred_iter[0][target_state] == np.max(cf_pred_iter[0]))
                    # cf_to_keep_iter = cf_pred_iter[0][init_state] <= 0.5 - self.margin

                    cost_iter, similarity_iter  = self.cf_cost(review, [new_reviews[k]], target_state)
                    #Création des arrêtes et noeuds du graph
                    print("edge " + str(i)+ "-" + str(indx) + ", state : " + str(cf_state_iter) + ", cf candidate: " + str(cf_to_keep_iter) + ", cost: " + str(cost_iter))
                    # print(' '.join(new_reviews_tokenized[k]))
            #       print(' '.join(new_review_tokenized))
                    print(str(old_token) + str(" -----> ") + str(new_reviews_tokenized[k][j]))
                    G_text.add_edge(i, indx)
                    G_text.add_node(indx, text = new_reviews_tokenized[k], 
                                    hist_mask = predecessor_hist_mask_iter, hist_mask_text = predecessor_text_masked_iter, 
                                    attrib_coeff = attribution_iter.loc[j]['Attribution coefficient'], cost = float(cost_iter), 
                                    similarity = similarity_iter, state = cf_state_iter, cf = cf_to_keep_iter)

                    wait_list = [(x, G_text.nodes.data()[x]['cost']) for x in G_text.nodes() if G_text.out_degree(x)==0 and G_text.in_degree(x)==1 and G_text.nodes.data()[x]['cf']==False]
                    wait_list.sort(reverse=False, key = lambda tup : tup[1])

                    #Plot de CF généré avec mise en avant des tokens changés     
                    if cf_to_keep_iter :
                        nb_cf+=1

                    #Si on a trouvé assez de cf ou bien si on a  changé tous les mots, on arrête
                    if (nb_cf == self.n) | (len(predecessor_text_masked_iter) == new_reviews_tokenized[k]) :
                        break
 
        #Viz cf détectés
        nodes_result = [x for x in G_text.nodes() if G_text.nodes.data()[x]['cf']]
        change_to_plot_html = []
        cf_list = []
        
        if len(nodes_result)>0 :
            for r in nodes_result :
            #     compute_attribution(text=cf_review, sentiment_model=sentiment_model, tokenizer=tokenizer, attribution=attribution)
                token_change = attribution_coeff.copy()
                token_change['Attribution coefficient'] = 0
                token_change['token'] = G_text.nodes.data()[r]['text']
                cf_list.append(' '.join(G_text.nodes.data()[r]['text']))
                cf_token_change = G_text.nodes.data()[r]['hist_mask']
                token_change.iloc[cf_token_change,1]=1
                change_to_plot_html.append(plot_change(token_change, n_colors=100))
            
           
        else :
            cf_list = []
            change_to_plot_html = []
            
        self.graph_cf.append(G_text)
        self.cf_list.append(cf_list)
        self.reviews.append(review)
        self.cf_html_list.append(change_to_plot_html)
        
        
        return(G_text, cf_list, change_to_plot_html)
    
    def bleu_score(self):
        """compute cf BLEU score

        Args:
        
        """
        
        if len(self.graph_cf) == 0 :
            raise Exception("No cf computed yet, please compute some cf first")
        
        avg_bleu_score_list = []
        for idx in range(len(self.reviews)) :
            init = [self.reviews[idx][0].split()]
            avg_bleu_score_iter_list = []
            for cf in self.cf_list[idx] :
                BLEUscore = nltk.translate.bleu_score.sentence_bleu(init, cf.split())
                avg_bleu_score_iter_list.append(BLEUscore)

            avg_bleu_score_list.append(np.mean(avg_bleu_score_iter_list))
            
        return(avg_bleu_score_list)
    
    def diversity(self, sentence_similarity:str="cls_embedding", reg_coeff = 1) :
        """compute dpp diversity & average distance between cf

        Args:
            sentence_similarity (str): text distance. Defaults to "cls_embedding".
            reg_coeff (int): regularization coefficient for dpp determinant. Defaults to 1.
        """
        
        if len(self.graph_cf) == 0 :
            raise Exception("No cf computed yet, please compute some cf first")
        
        det_list = []
        avg_dist_list = []
        
        for idx in range(len(self.graph_cf)) :
        
            nodes_result = [x for x in self.graph_cf[idx].nodes() if self.graph_cf[idx].nodes.data()[x]['cf']]
            nodes_result.append(0)

            dist_matrix = np.empty([len(nodes_result), len(nodes_result)], dtype=float)
            
            for i,j in enumerate(nodes_result) :
                init_review = [" ".join(self.graph_cf[idx].nodes.data()[j]['text'])]
                for k,l in enumerate(nodes_result) :
                    if j==l :
                        dist_matrix[i,k] = 1
                        pass
                    cf_review = [" ".join(self.graph_cf[idx].nodes.data()[l]['text'])]
                    
                    if sentence_similarity == "cls_embedding" :  
                        similarity = self.classifier.cls_similarity(init_review, cf_review)
                
                    if sentence_similarity == "sentence_transformer" :
                        similarity = self.sentence_transformer_similarity(init_review, cf_review)
              
                    dist_matrix[i,k] = similarity
        
            #computing dpp matrix & determinant
            dpp_mat = 1/(dist_matrix[1:, 1:] + reg_coeff)
            det = np.linalg.det(dpp_mat)
            det_list.append(det)
            #compute average distance between cf
            avg_dist = sum(dist_matrix[1:, 1:])/dist_matrix[1:, 1:].shape[0]
            avg_dist_list.append(avg_dist[0])
        
        return(det_list, avg_dist_list)

    def grammatical_accuracy(self, num_beams=5) :
        """assess the grammatical accuracy of the CF generated

        Args:
            num_beams (int, optional):exhaustivity of the paths computed. Defaults to 5.

        Raises:
            Exception: CFs have to be computed before
        """
   
        if len(self.graph_cf) == 0 :
            raise Exception("No cf computed yet, please compute some cf first")
        
        from happytransformer import HappyTextToText, TTSettings
        
        happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")  
        # args = TTSettings(num_beams, min_length=1)
        args = TTSettings(num_beams)
        
        grammar_accuracy_list = []
        
        for idx in range(len(self.cf_list)) :
            grammar_accuracy_iter = []
            for cf in self.cf_list[idx] :
                text_input = cf.capitalize()
                if text_input[-1] !='.':
                    text_input += '.'
                #computing grammar correction
                result = happy_tt.generate_text("grammar: " + text_input, args=args)
                if (text_input == result.text) :
                    grammar_accuracy_iter.append(1)
                else :
                    grammar_accuracy_iter.append(0)
                # print(result)
                
                #Add average grammar accuracy for one set of CF    
            grammar_accuracy_list.append(np.mean(grammar_accuracy_iter))
            
        return(grammar_accuracy_list)
                    
    def perplexity(self, stride=64) :
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast

        device = "cuda"
        model_id = "gpt2-large"
        model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
        tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
        
        text =  " \n\n ".join([" \n\n ".join(cf) for cf in self.cf_list])
        
        encodings = tokenizer(text, return_tensors="pt")
        max_length = model.config.n_positions

        nlls = []
        for i in range(0, encodings.input_ids.size(1), stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, encodings.input_ids.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs[0] * trg_len

            nlls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
        
        return(ppl)
    
    def loss(self, stride = 64):
        bleu_score = np.mean(self.bleu_score())
        t5_grammar = np.mean(self.grammatical_accuracy())
        #focusing on dpp diversity, not average distance
        diversity = np.mean(self.diversity(), axis=0)[0]
        # perplexity = self.perplexity(stride)
        
        loss = -(bleu_score + t5_grammar  + diversity)
        
        return(loss)
