# it was used this guide https://www.depends-on-the-definition.com/named-entity-recognition-with-bert/
import pandas as pd
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam
from pytorch_pretrained_bert import WordpieceTokenizer
from seqeval.metrics import f1_score

def conll_reader(inpt_file):
    sent = []
    lab = []
    f = open(inpt_file, "r")
    tok_lst = []
    label_lst = []
    for line in f:
        if line == '\t\n' or line == '\n':
            sent.append(tok_lst)
            lab.append(label_lst)
            tok_lst = []
            label_lst = []
            continue
        
        line = line.split()
        tok = line[0]
        try:
            label = line[-1]
        except:
            print(line, ix)
        tok_lst.append(tok)
        label_lst.append(label)
    f.close()
    
    return sent, lab

train_sent, labels = conll_reader('wnut17train.conll')
dev_sent, labels_dev = conll_reader('emerging.dev.conll')
test_sent, labels_test = conll_reader('emerging.test.annotated')
tags_vals = ['B-location',
     'B-creative-work',
     'I-location',
     'I-group',
     'I-corporation',
     'O',
     'B-corporation',
     'B-person',
     'I-person',
     'B-product',
     'I-creative-work',
     'I-product',
     'B-group',
     'X']
tag2idx = {t: i for i, t in enumerate(tags_vals)}

MAX_LEN = 75
bs = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
def data_tok(sent_data, lab_data):
    input_texts = []
    input_labels = []
    for sent, tags in zip(sent_data,lab_data):
        new_tags = []
        new_text = []
        for word, tag in zip(sent,tags):
            sub_words = tokenizer.wordpiece_tokenizer.tokenize(word)
            for count, sub_word in enumerate(sub_words):
                if count > 0:
                    tag = 'X'
                new_tags.append(tag)
                new_text.append(sub_word)
        input_texts.append(new_text)
        input_labels.append(new_tags)
    return input_texts, input_labels

tokenized_texts, labels = data_tok(train_sent, labels)

tokenized_dev, labels_dev = data_tok(dev_sent, labels_dev)

tokenized_test, labels_test = data_tok(test_sent, labels_test)
        
input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                     maxlen=MAX_LEN, value=tag2idx["O"], padding="post",
                     dtype="long", truncating="post")
attention_masks = [[float(i>0) for i in ii] for ii in input_ids]

input_dev = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_dev],
                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
tags_dev = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels_dev],
                     maxlen=MAX_LEN, value=tag2idx["O"], padding="post",
                     dtype="long", truncating="post")
attention_masks_dev = [[float(i>0) for i in ii] for ii in input_dev]

input_test = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_test],
                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
tags_test = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels_test],
                     maxlen=MAX_LEN, value=tag2idx["O"], padding="post",
                     dtype="long", truncating="post")
attention_masks_test = [[float(i>0) for i in ii] for ii in input_test]

tr_inputs = torch.tensor(input_ids)
val_inputs = torch.tensor(input_dev)
tr_tags = torch.tensor(tags)
val_tags = torch.tensor(tags_dev)
tr_masks = torch.tensor(attention_masks)
val_masks = torch.tensor(attention_masks_dev)
test_inputs = torch.tensor(input_test)
test_tags = torch.tensor(tags_test)
test_masks = torch.tensor(attention_masks_test)

train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)

test_data = TensorDataset(test_inputs, test_masks, test_tags)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=bs)

model = BertForTokenClassification.from_pretrained("bert-base-cased", num_labels=len(tag2idx))

#model.cuda()

FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters()) 
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

epochs = 5
max_grad_norm = 1.0

for _ in range(epochs):
    # TRAIN loop
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        # forward pass
        loss = model(b_input_ids, token_type_ids=None,
                     attention_mask=b_input_mask, labels=b_labels)
        # backward pass
        loss.backward()
        # track train loss
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        model.zero_grad()
    # print train loss per epoch
    print("Train loss: {}".format(tr_loss/nb_tr_steps))
    # VALIDATION on validation set
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():
            tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                                  attention_mask=b_input_mask, labels=b_labels)
            logits = model(b_input_ids, token_type_ids=None,
                           attention_mask=b_input_mask)
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.append(label_ids)
        
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        
        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy
        
        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1
    eval_loss = eval_loss/nb_eval_steps
    print("Validation loss: {}".format(eval_loss))
    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
    pred_tags = [tags_vals[p_i] for p in predictions for p_i in p]
    valid_tags = [tags_vals[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
    print("F1-Score: {}".format(f1_score(pred_tags, valid_tags)))


model.eval()
predictions = []
true_labels = []
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0
for batch in test_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch

    with torch.no_grad():
        tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                              attention_mask=b_input_mask, labels=b_labels)
        logits = model(b_input_ids, token_type_ids=None,
                       attention_mask=b_input_mask)
        
    logits = logits.detach().cpu().numpy()
    predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
    label_ids = b_labels.to('cpu').numpy()
    true_labels.append(label_ids)
    tmp_eval_accuracy = flat_accuracy(logits, label_ids)

    eval_loss += tmp_eval_loss.mean().item()
    eval_accuracy += tmp_eval_accuracy

    nb_eval_examples += b_input_ids.size(0)
    nb_eval_steps += 1

pred_tags = [[tags_vals[p_i] for p_i in p] for p in predictions]
valid_tags = [[tags_vals[l_ii] for l_ii in l_i] for l in true_labels for l_i in l ]
print("Validation loss: {}".format(eval_loss/nb_eval_steps))
print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))
