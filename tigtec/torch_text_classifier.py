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
#NLP/DL librarie
#Transformers
import transformers
from transformers import BertTokenizerFast, DistilBertModel, BertModel, DistilBertTokenizer, DistilBertForMaskedLM, DistilBertConfig
from transformers import AdamW, get_linear_schedule_with_warmup


#PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler



# Create the BertClassfier class
class BertClassifier(nn.Module):
    """DistillBert Model for Classification Tasks.
    """
    def __init__(self, nb_class, model='distilbert' , freeze_bert=False):
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

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
              nn.Linear(D_in, D_out)
#             nn.Linear(D_in, H),
#             nn.ReLU(),
#             #nn.Dropout(0.5),
#             nn.Linear(H, D_out)
        )

        # Freeze the DistillBERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
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
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits
    
    def predict(self, text, tokenizer=tokenizer) :
        input, mask = preprocessing_for_bert(text, tokenizer)

        with torch.no_grad():
                    logits = self(input, mask)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        
        return(probs)

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
    return(y)

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
    y = categorical_variables(target)
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
    
    return train_dataloader, test_dataloader
    

def initialize_model(train_dataloader, nb_class, epochs=4):
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    # Instantiate Bert Classifier
    bert_classifier = BertClassifier(nb_class, freeze_bert=False)

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