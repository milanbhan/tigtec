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
from transformers import DistilBertTokenizer

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
        else :
            self.sentence_transformer = None
        self.topk = topk
        self.mask_variety = mask_variety
        self.margin = margin
        self.beam_width = beam_width
        self.alpha = alpha
        
        if self.classifier.bert.name_or_path == 'textattack/bert-base-uncased-imdb':
            self.tokenizer_mask = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        
        #List of cf & cf informations
        self.graph_cf = []
        self.cf_list = []
        self.cf_html_list = []
        self.reviews = []
        
    def mlm_inference(self, masked_text) :
        if self.classifier.bert.name_or_path == 'textattack/bert-base-uncased-imdb':
            inputs = self.tokenizer_mask(masked_text, return_tensors='pt')
        else :     
            inputs = self.classifier.tokenizer(masked_text, return_tensors='pt')
  
        with torch.no_grad():
            logits = self.mlm(**inputs).logits

        # retrieve index of [MASK]
        mask_token_index = (inputs.input_ids == self.classifier.tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

        #   predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        logit  = logits[0, mask_token_index]
        predicted_tokens_id = torch.topk(logit.flatten(), self.topk).indices
        words = self.classifier.tokenizer.decode(predicted_tokens_id)
        words = words.split(" ")
        punct_remove_list = ["#", ",", ";", "!", "?", "'", ".", ".", "-", "&", ")", "(", '"', "/", "@", "’"]
        
        for word in words.copy() :
            if any(punct in word for punct in punct_remove_list):
                words.remove(word)
        
        # for word in words :
        #     if (word[-1] in punct_remove_list) | (word[0] in punct_remove_list) :
        #         words.remove(word)
        #     else : 
        #         pass
        return(words)
    
    def set_graph_cf(self, graph_cf) :
        self.graph_cf = graph_cf
        
    def set_cf_list(self, cf_list) :
        self.cf_list = cf_list
        
    def set_reviews(self, reviews) :
        self.reviews = reviews
    
    def sentence_transformer_similarity(self, text1, text2, sentence_transformer = None) :
        sentences = text1 + text2
        
        if self.sentence_transformer == None :
            if sentence_transformer == None :
                raise Exception("Sorry, no Sentence transformer in the class or in the parameters")
            embeddings = sentence_transformer.encode(sentences)
            similarity = 1 - spatial.distance.cosine(embeddings[0], embeddings[1])
        else :
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
        # print(review)
        # print(to_mask)
        token_max = review[to_mask]
        new_review = review.copy()
        new_review[to_mask] = self.classifier.tokenizer.mask_token


        
        #Constitution de la nouvelle review avec le mask, et MLM inférence pour remplacement du mask par le nouveau token
        new_review = ' '.join(new_review)
        new_tokens = self.mlm_inference(masked_text=[new_review])
        # print([new_review])
        # print(new_tokens)
        
        #si la MLM ne renvoie rien, on refait tourner le MLM en filtrant au-delà de 4 tokens de + que le mask
        if len(new_tokens) == 0 :
            new_review = review.copy()
            new_review[to_mask] = self.classifier.tokenizer.mask_token
            new_review = new_review[0:to_mask + 4]
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
    
    def get_tigtec_cf_token_importance(self, idx):
        
        #correction coût initial trop élevé (à corriger à terme directement dans le code de génération de cf)
        if self.graph_cf[idx].nodes.data()[0]['cost']>=1:
            self.graph_cf[idx].nodes.data()[0]['cost']= (self.graph_cf[idx].nodes.data()[0]['cost'] - 1)-self.alpha
        #initialisation vecteur de token importance
        loss_importance = len(self.graph_cf[idx].nodes.data()[0]['text']) * [0]
        nodes_result = [x for x in self.graph_cf[idx].nodes() if self.graph_cf[idx].nodes.data()[x]['cf']]
        #reconstitution du chemin entre cf et initial pour décomposer l'évol de la loss
        paths = list(nx.all_simple_paths(self.graph_cf[idx], source=0, target=nodes_result[0]))[0]
        for k, l in reversed(list(enumerate(paths))):
            if k==0:
                break
            else:
                delta_loss = self.graph_cf[idx].nodes.data()[l]['cost'] - self.graph_cf[idx].nodes.data()[paths[k-1]]['cost']
                id_token = self.graph_cf[idx].nodes.data()[l]['hist_mask'][-1]
                loss_importance[id_token] = np.abs(delta_loss)
        #Création df output
        tokens = self.graph_cf[idx].nodes.data()[0]['text']
        token_importance = loss_importance
        attribution_coefficient=pd.DataFrame({"token":tokens,"Attribution coefficient":token_importance})
        return(attribution_coefficient)

    
    def generate_cf(self, review, target, indx_max = 1000, base=None, cf=None, idx=None):
      #Prédictions text initial
        init_pred = self.classifier.predict(review)
        nb_class = init_pred.shape[1]
        init_state = np.argmax(init_pred)
        target_state = target
        method = self.attribution
        return_node=False
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
        
        #Initialisation du graph basé sur le text initial. Si cf_init autre que None, alors tigtec specific

        attribution_coeff_init = self.classifier.compute_token_importance(text=cf_review, method = method, base=base, cf=cf, idx=idx)
        attribution_coeff = attribution_coeff_init.copy()

            
        text_initial_tokenized = attribution_coeff['token'].tolist()
        #   text_initial_tokenized = [tokenizer.decode(t).replace(" ", "") for t in token_list_encoded]
        G_text = nx.DiGraph()
        G_text.add_node(0, text = text_initial_tokenized, hist_mask = [], hist_mask_text = [], attrib_coeff = 1, cost = init_cost, similarity = 1, state=init_state, cf = False)
        wait_list = [(0,1)]
        indx=0
        
        #Premier test : on itère jusqu'à la profondeur max en monde beamsearch
        #   for depth in range(len(text_initial_tokenized)) :
        nb_cf = 0  
        while (nb_cf < self.n) & (indx <= indx_max) :
            if len(wait_list)==0:
                predecessor_hist_mask = G_text.nodes.data()[i]['hist_mask']
                predecessor_text_masked = G_text.nodes.data()[i]['hist_mask_text']
                predecessor_text = G_text.nodes.data()[i]['text']
                return_node = True
            else:
                i = wait_list[0][0]
                #Si trop long, on abandonne

                #Récupération historique des tokens masqués et du text du noeud parent
                predecessor_hist_mask = G_text.nodes.data()[i]['hist_mask']
                predecessor_text_masked = G_text.nodes.data()[i]['hist_mask_text']
                predecessor_text = G_text.nodes.data()[i]['text']
            
            if (len(text_initial_tokenized) == len(predecessor_hist_mask)) :
                break
            
            if self.explo_strategy == 'evolutive' :
                attribution_coeff = self.classifier.compute_token_importance(text=[' '.join(predecessor_text)], method = method, base=base)
            
            #On filtre l'attribution en enlevant les tokens déjà masqués/remplacés
            attribution_iter = attribution_coeff[attribution_coeff.index.isin(predecessor_hist_mask)==False]
            #to do : penser au cas de figure avec des attribution égales
            #
            if return_node:
                nodes_result = [x for x in G_text.nodes() if G_text.nodes.data()[x]['cf']]
                nb_cf_found = len(nodes_result)
                ind_to_mask = attribution_iter[attribution_iter['token'].isin([".", ",", ";"])==False].nlargest(self.beam_width + nb_cf_found , 'Attribution coefficient')['token'].index.tolist()[nb_cf_found-1:]

            else:
                ind_to_mask = attribution_iter[attribution_iter['token'].isin([".", ",", ";"])==False].nlargest(self.beam_width, 'Attribution coefficient')['token'].index.tolist()
            # print(predecessor_hist_mask)
            # print(wait_list)
            #Pour chaque token à changer, chacun dans un nouveau noeud
            for j in ind_to_mask :
                if ((nb_cf == self.n) | (indx > indx_max)) :
                    break
                predecessor_hist_mask_iter = predecessor_hist_mask.copy()
                predecessor_hist_mask_iter.append(j)
                predecessor_text_masked_iter = predecessor_text_masked.copy()
                text_iter = predecessor_text.copy()

                new_reviews, new_reviews_tokenized, old_token, new_tokens = self.replace_token(text_iter,j, target_state)

                #Ajout à l'historique des tokens masqués
                predecessor_text_masked_iter.append(old_token)
                for k in range(len(new_reviews)) :
                    indx+=1
                    if ((nb_cf == self.n) | (indx > indx_max)) :
                        break
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

                    
                    # print(wait_list)

                    #Plot de CF généré avec mise en avant des tokens changés     
                    if cf_to_keep_iter :
                        nb_cf+=1

                    #Si on a trouvé assez de cf ou bien si on a  changé tous les mots, on arrête
                    if (nb_cf == self.n) | (len(predecessor_text_masked_iter) == new_reviews_tokenized[k]) :
                        break
                    
                    wait_list = [(x, G_text.nodes.data()[x]['cost']) for x in G_text.nodes() if G_text.out_degree(x)==0 and G_text.in_degree(x)==1 and G_text.nodes.data()[x]['cf']==False]
                    wait_list.sort(reverse=False, key = lambda tup : tup[1])
 
        #Viz cf détectés
        nodes_result = [x for x in G_text.nodes() if G_text.nodes.data()[x]['cf']]
        change_to_plot_html = []
        cf_list = []
        
        if len(nodes_result)>0 :
            for r in nodes_result :
            #     compute_attribution(text=cf_review, sentiment_model=sentiment_model, tokenizer=tokenizer, attribution=attribution)
                token_change = attribution_coeff_init.copy()
                token_change['Attribution coefficient'] = 0
                try :
                    token_change['token'] = G_text.nodes.data()[r]['text']
                    cf_list.append(' '.join(G_text.nodes.data()[r]['text']))
                    cf_token_change = G_text.nodes.data()[r]['hist_mask']
                    token_change.iloc[cf_token_change,1]=1
                    change_to_plot_html.append(plot_change(token_change, n_colors=100))
                except :
                    print("exception:  ")
                    print([t for t in token_change['token']])
                    print(G_text.nodes.data()[r]['text'])
                
            
           
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
    
    def text_similarity(self, sentence_similarity = "sentence_transformer", sentence_transformer = None) :
        """ sentence_similarity (str, optional): _description_. Defaults to "sentence_transformer".
        """
        similarity_list= []
        for idx in range(len(self.graph_cf)) :
            similarity_iter = []
            init_review =  [' '.join(self.graph_cf[idx].nodes.data()[0]['text'])]
            cf_nodes = [x for x in self.graph_cf[idx].nodes() if self.graph_cf[idx].nodes.data()[x]['cf']]
            for n in cf_nodes :
                cf_review = [' '.join(self.graph_cf[idx].nodes.data()[n]['text'])]
                if sentence_similarity == "cls_embedding" :
                    similarity = self.classifier.cls_similarity(init_review, cf_review)
                    similarity_iter.append(similarity)
                else :
                    similarity = self.sentence_transformer_similarity(init_review, cf_review, sentence_transformer)
                    similarity_iter.append(similarity)
            similarity_list.append(np.mean(similarity_iter))
        return(similarity_list)

    def token_change_rate(self) :
        """compute token change rate : number of cf token change over total number of tokens

        Args:
            None
        """

        token_cr_list = []
        if len(self.graph_cf) == 0 :
            raise Exception("Sorry, no cf computed yet")
        
        for idx in range(len(self.graph_cf)) :
            token_cr_iter = []
            cf_nodes = [x for x in self.graph_cf[idx].nodes() if self.graph_cf[idx].nodes.data()[x]['cf']]
            for n in cf_nodes :
                cr = len(self.graph_cf[idx].nodes.data()[n]['hist_mask']) / len(self.graph_cf[idx].nodes.data()[n]['text'])
                token_cr_iter.append(cr)
            token_cr_list.append(np.mean(token_cr_iter))
        return(token_cr_list)
    
    def success_rate(self) :
        """compute success rate : number of cf founded over number of cf targeted

        Args:
            None
        """

        success_rate_list = []
        for idx in range(len(self.graph_cf)) :
            cf_nodes = [x for x in self.graph_cf[idx].nodes() if self.graph_cf[idx].nodes.data()[x]['cf']]
            success_rate = len(cf_nodes) / self.n
            success_rate_list.append(success_rate)
        return(success_rate_list)
    
    def diversity(self, reg_coeff = 1) :
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
                    
                    if self.sentence_similarity == "cls_embedding" :  
                        similarity = self.classifier.cls_similarity(init_review, cf_review)
                
                    else:
                        if self.sentence_transformer == None :
                            self.sentence_transformer = SentenceTransformer('paraphrase-MiniLM-L6-v2')
                        else :
                            pass
                        similarity = self.sentence_transformer_similarity(init_review, cf_review)
              
                    dist_matrix[i,k] = (1-similarity)/2
        
            #computing dpp matrix & determinant
            dpp_mat = 1/(dist_matrix[1:, 1:] + reg_coeff)
            det = np.linalg.det(dpp_mat)
            det_list.append(det)
            # #compute average distance between cf
            # avg_dist = sum(dist_matrix[1:, 1:])/dist_matrix[1:, 1:].shape[0]
            # avg_dist_list.append(avg_dist[0])
        return(det_list)        
        # return(det_list, avg_dist_list)
        #test modif

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
        ppl_list = []
        for idx in range(len(self.cf_list)) :
        
            text =  " \n\n ".join([" \n\n ".join(cf) for cf in self.cf_list[idx]])
            if len(text) == 0 :
                continue     
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
            print(str(idx) + " : " + str(float(ppl)))
            ppl_list.append(ppl)
        
        return([p.cpu().numpy() for p in ppl_list])
    
    def loss(self, stride = 64):
        bleu_score = np.mean(self.bleu_score())
        t5_grammar = np.mean(self.grammatical_accuracy())
        #focusing on dpp diversity, not average distance
        diversity = np.mean(self.diversity(), axis=0)[0]
        # perplexity = self.perplexity(stride)
        
        loss = -(bleu_score + t5_grammar  + diversity)
        
        return(loss)
    
def boost_cf(cf, n, targets, indx_max, cf_ti_method='cf_token_importance'):
    if len(cf.cf_list)==0:
        raise Exception("No counterfactual already computed. Please first indicate some counterfactual examples")
    else:
        cf_enhancer = tigtec(classifier = cf.classifier,
            mlm = cf.mlm,
            n = n,
            attribution = cf_ti_method,
            explo_strategy = 'static',
            sentence_similarity = cf.sentence_similarity,
            topk = cf.topk,
            mask_variety = cf.mask_variety,
            margin = cf.margin,
            beam_width = 1,
            alpha = cf.alpha)
        
        for i,j in enumerate(cf.reviews):
            print(i)
            if len(cf.cf_list[i])==0:
                cf_enhancer.cf_list.append([])
                cf_enhancer.graph_cf.append(cf.graph_cf[i])

            else:
                nodes_result = [x for x in cf.graph_cf[i].nodes() if cf.graph_cf[i].nodes.data()[x]['cf']]
                min_nodes = min(nodes_result)
                indx_max = len(cf.graph_cf[i].nodes().data()[min_nodes]['hist_mask']) * cf.mask_variety
                
                if cf_ti_method=='cf_token_importance':
                    cf_enhancer.generate_cf(j, target = targets[i], indx_max=indx_max, base=[cf.cf_list[i][0]])
                elif cf_ti_method=='tigtec_cf_token_importance':
                    cf_enhancer.generate_cf(j, target = targets[i], indx_max=indx_max, base=[cf.cf_list[i][0]], cf=cf, idx=i)
                if len(cf_enhancer.cf_list[i])< len(nodes_result):
                    print(str(i) + " not found or not sufficient")
                    cf_enhancer.cf_list[i] = cf.cf_list[i]
                    cf_enhancer.graph_cf[i] = cf.graph_cf[i]
                    # cf_enhancer.cf_html_list[-1] = cf.cf_html_list[i]
                    
    return(cf_enhancer)
            
            
             
