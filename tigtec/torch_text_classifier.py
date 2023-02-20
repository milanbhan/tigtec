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
from sklearn.model_selection import train_test_split
import seaborn as sns
import random
import time
#for coloring text
from scipy.special import softmax
from IPython.core.display import display, HTML
import seaborn as sns
 
#NLP/DL librarie
#Transformers
import transformers
from transformers import BertTokenizerFast, DistilBertModel, BertModel, DistilBertTokenizer, DistilBertForMaskedLM, DistilBertConfig, FlaubertTokenizer, FlaubertModel, CamembertTokenizer, CamembertModel, CamembertForMaskedLM, AutoModelForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup


#PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

#XAI libraries
import shap
import lime
from lime.lime_text import LimeTextExplainer
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz
#graph library
import networkx as nx

import nltk


# Create the BertClassfier class
class BertClassifier(nn.Module):
    """DistillBert Model for Classification Tasks.
    """
    def __init__(self, nb_class, tokenizer, model='distilbert' , freeze_bert=False, max_len = 68):
        """
        @param    model: a BertModel object : DistilBertModel or BERTModel
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 50, nb_class

        if torch.cuda.is_available():       
            self.device = torch.device("cuda")
            print(f'There are {torch.cuda.device_count()} GPU(s) available.')
            print('Device name:', torch.cuda.get_device_name(0))

        else:
            print('No GPU available, using the CPU instead.')
            self.device = torch.device("cpu")
        # Instantiate BERT model
        if model == 'distilbert' :
            self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased',
                                                    output_attentions = True, 
                                                    return_dict = True,
                                                    output_hidden_states = False)
        elif model == 'bert' :
            self.bert = BertModel.from_pretrained('bert-base-uncased',
                                                    output_attentions = True, 
                                                    return_dict = True,
                                                    output_hidden_states = False)
            
        elif model == 'flaubert' :
            self.bert = FlaubertModel.from_pretrained("flaubert/flaubert_base_cased",
                                                    output_attentions = True, 
                                                    return_dict = True,
                                                    output_hidden_states = False)
        elif model == 'camembert' :
            self.bert = CamembertModel.from_pretrained("camembert-base",
                                                    output_attentions = True, 
                                                    return_dict = True,
                                                    output_hidden_states = False)
        elif model == 'BERT_text_attack':
            # self = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
            sentiment_model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb",
                                                    output_attentions = True, 
                                                    return_dict = True,
                                                    output_hidden_states = False)
            self.bert = sentiment_model.bert
            self.classifier = sentiment_model.classifier
        
        self.tokenizer = tokenizer
        self.true_to_pred = {}
        self.pred_to_true = {}
        self.max_len = max_len
        
        
        # Instantiate an one-layer feed-forward classifier
        
        if model != 'BERT_text_attack':
            self.classifier = nn.Sequential(
                nn.Linear(D_in, D_out)
    #             nn.Linear(D_in, H),
    #             nn.ReLU(),
    #             #nn.Dropout(0.5),
    #             nn.Linear(H, D_out)
            )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
    def predict(self, text) :
        input, mask = preprocessing_for_bert(text, self.tokenizer)

        with torch.no_grad():
            logits = self(input, mask)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        
        return(probs)
    
    
    def random_token_importance(self, text):
        token_list_encoded = [t for t in preprocessing_for_bert(text, self.tokenizer, self.max_len)[0].tolist()[0] if t not in [101, 102, 103]]
        token_list_encoded
        token_list = [self.tokenizer.decode(t).replace(" ", "") for t in token_list_encoded]
        random_list = [random.uniform(0, 1) for t in token_list_encoded]
        sum_random_list = sum(random_list)
        random_list = [t / sum_random_list for t in random_list ]

        attribution_coefficient = pd.DataFrame(columns = ["token", "Attribution coefficient"])
        attribution_coefficient['token'] = token_list
        attribution_coefficient['Attribution coefficient'] = random_list
        attribution_coefficient = attribution_coefficient[attribution_coefficient.token != '[PAD]']
        
        return(attribution_coefficient)
        
    def intergrated_gradient_token_importance(self, text):
        # inputs = self.tokenizer.batch_encode_plus(
        #     text,  # Preprocess sentence,
        #     truncation=True,
        #     add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
        #     max_length=self.max_len,                  # Max length to truncate/pad
        #     padding='max_length',         # Pad sentence to max length
        #     #return_tensors='pt',           # Return PyTorch tensor
        #     )
        
        # # Convert lists to tensors
        # tokenized=torch.tensor(inputs["input_ids"])
        # attention_mask=torch.tensor(inputs["attention_mask"])
            
        layer = self.bert.embeddings
        
        def ig_forward(inputs):
            return(self.bert(inputs["input_ids"], inputs["attention_mask"]).logits)
        
        ig = LayerIntegratedGradients(self.forward, layer)
        true_class = np.argmax(self.predict(text))
        input_ids, base_ids = ig_encodings(self.tokenizer, text)
        attrs, delta = ig.attribute(inputs["input_ids"], inputs["attention_mask"], base_ids, target=true_class, return_convergence_delta=True)
        scores = attrs.sum(dim=-1)
        scores = (scores - scores.mean()) / scores.norm()
        return(scores)

    
    def lime_token_importance(self, text):
        token_list_encoded = [t for t in preprocessing_for_bert(text, self.tokenizer, self.max_len)[0].tolist()[0] if t not in [0,101, 102, 103]]
        token_list_encoded
        token_list = [self.tokenizer.decode(t).replace(" ", "") for t in token_list_encoded]
        text_lime = ' '.join(token_list)
        text_lime = text_lime.replace(" ##", "")
        
        #Declare LIME explainer
        label_names = ["negative", "positive"]
        explainer = LimeTextExplainer(class_names=label_names)
        num_token_compute = int(np.round(len(text_lime.split(" "))/2))

        #Compute lime coeff
        exp = explainer.explain_instance(text_lime, self.predict, num_features=num_token_compute)
        
        #Prepare attribution coeff dataframe
        tokens_df = pd.DataFrame({"token":text_lime.split(" ")}).reset_index()
    #       tokens_df = pd.DataFrame({"token":text.split(" ")}).reset_index()
        lime_coeff = pd.DataFrame(exp.as_list())
        attribution_coefficient = tokens_df.merge(lime_coeff, how='left', left_on='token', right_on=0).drop(columns=0).rename(columns={1 : "Attribution coefficient"})
        attribution_coefficient['Attribution coefficient'].loc[attribution_coefficient['Attribution coefficient'].isnull()]=0
        attribution_coefficient['Attribution coefficient'] = np.abs(attribution_coefficient['Attribution coefficient'])
        
        return(attribution_coefficient)
    
    def attention_token_importance(self, text):
        input, mask = preprocessing_for_bert(text, self.tokenizer, self.max_len)
        encoded_att = self.bert(input,attention_mask =mask)
        last_attention=encoded_att.attentions[-1]

        tokens,attentions = [], []
        #si le model ne classifie pas avec le token de classification lors du forward, on fait la moyenne des coeff de la dernière couche
        if self.bert.name_or_path == 'textattack/bert-base-uncased-imdb' :
            last_attention_mean = last_attention[0].mean(axis=0).mean(axis=0)
            for i, elt in enumerate(input[0]):
                if elt.numpy() != self.tokenizer.pad_token_id:
                    att = last_attention_mean[i]
                    tokens.append(self.tokenizer.decode([elt]) + '_' + str(i))
                    attentions.append(att.detach().numpy())
            attention_all_head=pd.DataFrame({"Token":tokens,"Attribution coefficient":attentions})
            # attention_all_head_mean = attention_all_head.groupby("Token").agg('mean').reset_index()
            attention_all_head['id'] = attention_all_head['Token'].apply(lambda t : int(t.split("_")[1]))
            attention_all_head['token'] = attention_all_head['Token'].apply(lambda t : t.split("_")[0])
            attention_all_head = (attention_all_head.sort_values("id")).reset_index(drop=True)
            attention_all_head['Attribution coefficient'] = attention_all_head['Attribution coefficient'].astype('float')
            attribution_coefficient = attention_all_head

        else :
            for head in range(0,12) :
                for i, elt in enumerate(input[0]):
                    #Ne pas prendre les éléments masqués
                    #!=0 car on ne prend pas les tokens padding

                    if elt.numpy() != self.tokenizer.pad_token_id:
                        #Sélection du coefficient d'attention associé,
                        #premier coefficient 0 pour ids1
                        #Dernier coefficient 0 = indice de l'objet de la classif par défault 
                        att = last_attention[0,head][0][i].item()

                        tokens.append(self.tokenizer.decode([elt]) + '_' + str(i))
                        attentions.append(att)

            attention_all_head=pd.DataFrame({"Token":tokens,"Attribution coefficient":attentions})
            attention_all_head_mean = attention_all_head.groupby("Token").agg('mean').reset_index()
            attention_all_head_mean['id'] = attention_all_head_mean['Token'].apply(lambda t : int(t.split("_")[1]))
            attention_all_head_mean['token'] = attention_all_head_mean['Token'].apply(lambda t : t.split("_")[0])
            attention_all_head_mean = (attention_all_head_mean.sort_values("id")).reset_index(drop=True)
            attribution_coefficient = attention_all_head_mean
        
        return(attribution_coefficient)
    
    def shap_token_importance(self, text):
        token_list_encoded = [t for t in preprocessing_for_bert(text, self.tokenizer, self.max_len)[0].tolist()[0] if t not in [101, 102, 103]]
        token_list_encoded
        token_list = [self.tokenizer.decode(t).replace(" ", "") for t in token_list_encoded]
        text_shap = ' '.join(token_list)
        text_shap = text_shap.replace(" ##", "")

        def f(review) :
            val = self.predict(review)[:,1]
            return val

        # build an explainer using a token masker
        explainer = shap.Explainer(f, self.tokenizer)
        shap_values = explainer([text_shap], fixed_context=1)
        
        attribution_coefficient = pd.DataFrame(columns = ["token", "Attribution coefficient"])
        attribution_coefficient["Attribution coefficient"]=np.abs(shap_values.values.tolist()[0])
    #       attribution_coefficient["token"]=shap_values.data[0].tolist()
        attribution_coefficient["token"]=[''] + token_list + ['']
        attribution_coefficient = attribution_coefficient[attribution_coefficient["token"]!='']
        
        return(attribution_coefficient)
    
    def compute_token_importance(self, text, method:str="attention"):
        if method == 'random' :
            attribution_coefficient = self.random_token_importance(text)
        if method == 'lime' :
            attribution_coefficient = self.lime_token_importance(text)
        if method == 'attention' :
            attribution_coefficient = self.attention_token_importance(text) 
        if method == 'shap' :
            attribution_coefficient = self.shap_token_importance(text)
        
        #Handling byte pair encoding without ##
        if self.tokenizer.name_or_path == 'camembert-base' :
            from nltk.tokenize import WordPunctTokenizer
            merged_tokenized = WordPunctTokenizer().tokenize(text[0])
            attribution_coefficient=attribution_coefficient[attribution_coefficient['token']!='']
            attribution_coefficient.reset_index(inplace=True, drop=True)
            for i in range(attribution_coefficient.shape[0]-1,0, -1) :
                if attribution_coefficient.iloc[i]['token'] in (merged_tokenized):
                    pass
                else :
                    attribution_coefficient['token'][i-1] += attribution_coefficient['token'][i]
                    attribution_coefficient['Attribution coefficient'][i-1] += attribution_coefficient['Attribution coefficient'][i]
            attribution_coefficient = attribution_coefficient[attribution_coefficient.token.isin(merged_tokenized)]
            attribution_coefficient = attribution_coefficient[attribution_coefficient.token.isin(merged_tokenized)]
            
        else : 
            attribution_coefficient['to_keep']='yes'
            attribution_coefficient['to_keep'][attribution_coefficient.token.str.startswith("##")]='no'
        
            #Regroupement des tokens séparés par le tokenizer, visibles grâce à ##
            for i in range(attribution_coefficient.shape[0]-1, 0, -1) :
                if attribution_coefficient.token[i][0:2]=="##" :
                    attribution_coefficient.token[i-1]+=attribution_coefficient.token[i]
                    attribution_coefficient['Attribution coefficient'][i-1]+=attribution_coefficient['Attribution coefficient'][i]
                    
            attribution_coefficient = attribution_coefficient[attribution_coefficient['to_keep']=='yes']
            attribution_coefficient.token = attribution_coefficient.token.str.replace("##", "")
            attribution_coefficient = attribution_coefficient[attribution_coefficient['token'].isin(['[CLS]', '[SEP]', '[PAD]'])==False]
        attribution_coefficient = attribution_coefficient[["token", "Attribution coefficient"]].reset_index(drop=True)
        
        attribution_coefficient['Attribution coefficient'][attribution_coefficient['token'].isin(['.', ',', ';', '!', '?', "'", ":", "’", ";,"])==True]=0
        
        return(attribution_coefficient)
        
    def sentence_embedding_cls(self, text) :
        input, mask = preprocessing_for_bert(text, self.tokenizer)
        encoded_att = self.bert(input,attention_mask=mask, output_hidden_states=True)
        #   embed_text = encoded_att[0][0][0]
        last_hidden_states  = encoded_att.hidden_states[-1]
        embed_text = last_hidden_states[0,0,:]

        return(embed_text)
    
    def cls_similarity(self, text1, text2) :
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        embed_text1 = self.sentence_embedding_cls(text1)
        embed_text2 = self.sentence_embedding_cls(text2)
        
        similarity = cos(embed_text1.unsqueeze(0), embed_text2.unsqueeze(0)).item()
        
        return(similarity)

    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        
        
        if self.bert.name_or_path == 'textattack/bert-base-uncased-imdb' :
            last_hidden_state_cls = outputs[1]
        else :
            last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits
    
    def plot_token_importance(self, text, method='attention', n_colors=15) :
          
        token_importance = self.compute_token_importance(text=text, method=method)
        
        pal = get_palette(theme="Reds", n_colors=n_colors)
        sentence=list(token_importance['token'].values)
        scores = list(token_importance['Attribution coefficient'].values)
        
        #   scores -= min(scores)
        scores = list(scores/(max(scores))) 
        scores = np.power(scores, 2)
        #   scores = softmax(scores)

        html = color_sentence(sentence, scores, pal, n_colors)
        return(html)
        
    

#Target variable encoding
def categorical_variables(labels):
    """encoding the target variable into numeric 

    Args:
        labels (pandas serie): target variable
    """
    y = np.array(labels)
    unique = np.unique(y)
    
    n_classes = len(unique)
    
    true_to_pred = {}
    pred_to_true = {}
    
    for i, elt in enumerate(unique):
        true_to_pred[elt] = i
        pred_to_true[i] = elt
    
    y1 = []
    for elt in y:
        y1.append(true_to_pred[elt])
    
    y = y1
    return(y,true_to_pred ,pred_to_true)

def preprocessing_for_bert(text, tokenizer, max_len = 68):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
#     input_ids = []
#     attention_masks = []

    inputs = tokenizer.batch_encode_plus(
            text,  # Preprocess sentence,
            truncation=True,
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=max_len,                  # Max length to truncate/pad
            padding='max_length',         # Pad sentence to max length
            #return_tensors='pt',           # Return PyTorch tensor
            )
        
     # Convert lists to tensors
    tokenized=torch.tensor(inputs["input_ids"])
    attention_mask=torch.tensor(inputs["attention_mask"])

    return tokenized, attention_mask

def build_data_loader(text, target, tokenizer, max_len = 68, batch_size = 32):
    """_summary_

    Args:
        text (_type_): _description_
        target (_type_): _description_
        tokenizer (_type_): _description_
        max_len (int, optional): _description_. Defaults to 68.
        batch_size (int, optional):For fine-tuning BERT, the authors recommend a batch size of 16 or 32. Defaults to 32.

    Returns:
        _type_: _description_
    """
    random_seed = 42
    #Preparing features
    X, X_mask = preprocessing_for_bert(text, tokenizer, max_len)
    
        #Preparing label 
    y,true_to_pred ,pred_to_true = categorical_variables(target)
    y = torch.tensor(y)

    # y = np_utils.to_categorical(y)

    ids_train, ids_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)
    masks_train, masks_test, _, _ = train_test_split(X_mask, y, test_size=0.2, random_state=random_seed) 

    X_train = [ids_train, masks_train]
    X_test = [ids_test, masks_test]


    # Create the DataLoader for our training set
    train_data = TensorDataset(ids_train, masks_train, y_train)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create the DataLoader for our validation set
    test_data = TensorDataset(ids_test, masks_test, y_test)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    
    return train_dataloader, test_dataloader, true_to_pred, pred_to_true
    

def initialize_model(train_dataloader, model, tokenizer, nb_class, true_to_pred, pred_to_true, epochs=4):
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    # Instantiate Bert Classifier
    bert_classifier = BertClassifier(nb_class, tokenizer, model, freeze_bert=False)
    bert_classifier.true_to_pred = true_to_pred
    bert_classifier.pred_to_true = pred_to_true

    # Tell PyTorch to run the model on GPU
    bert_classifier.to(bert_classifier.device)

    # Create the optimizer
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=5e-5,    # Default learning rate
                      eps=1e-8    # Default epsilon value
                      )

    # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def train(model, train_dataloader, optimizer, scheduler, test_dataloader=None, epochs=4, evaluation=False, loss_fn = nn.CrossEntropyLoss()):
    """Train the BertClassifier model.
    """
    # Start training loop
    print("Start training...\n")
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()
        
        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(model.device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids, b_attn_mask)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        print("-"*70)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy = evaluate(model, test_dataloader)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)
        print("\n")
    
    print("Training complete!")


def evaluate(model, val_dataloader, loss_fn = nn.CrossEntropyLoss()):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(model.device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        # Compute loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy

def ig_encodings(tokenizer, text):
    """Function to process text in order to compute integrated gradient

    Args:
        tokenizer (_type_): transformer tokenizer
        text (_type_): text to explain

    Returns:
        _type_: _description_
    """
    pad_id = tokenizer.pad_token_id
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    input_ids = tokenizer.encode(text, add_special_tokens=True)
    base_ids = [pad_id] * len(input_ids)
    base_ids[0] =  cls_id
    base_ids[-1] = sep_id
    return torch.LongTensor([input_ids]), torch.LongTensor([base_ids])


def get_palette(theme="YlOrBr", n_colors=1000):
    pal = sns.color_palette(theme, as_cmap=False, n_colors=n_colors)
#     pal = sns.color_palette("vlag", as_cmap=True)
    pal = (np.array(pal) * 255).astype(int)
    pal = list(map(lambda x: "#%02x%02x%02x" % tuple(x), pal))
    # pal = list(map(pal, lambda x: "#%02x%02x%02x" % tuple(x)))
    return np.array(pal)

def get_color(word, palette, score = 0.5, n_colors = 1000):
    # Score must be between zero and one
    index = max([int(n_colors * score - 1), 0])
    col = palette[index]
#     col = palette[int(n_colors * score - 1)]
    color = 'white' if score > 0.6 else 'black'
    return f'<a style="background-color : {col};color:{color};">{word}</a>'

def color_sentence(tokens, scores, palette, n_colors = 1000):
    html = [get_color(tokens[i], palette, scores[i], n_colors) for i, elt in enumerate(tokens)]
    html = '<div display:inline-block>' + ' '.join(html) + '</div>'
    return html

def plot_change(change_df, n_colors=2) :
      
  pal = get_palette(theme="Blues", n_colors=n_colors)

  sentence=list(change_df['token'].values)
  scores = list(change_df['Attribution coefficient'].values)
  
  html = color_sentence(sentence, scores, pal, n_colors)
  return(html)

