import pickle
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, BertForTokenClassification, LongformerForTokenClassification, Trainer, TrainingArguments
from tqdm import tqdm
import torch
import numpy as np
import utils



import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--split", help="determine how to split the article(sent,chunk,nosplit)", type=str,default="chunk")
parser.add_argument("--process", help="determine training mode(train,test)", type=str,default="test")
parser.add_argument("--checkpoint", help="meaning the path of the pre-trained model", type=str,default="model/bert-base-uncased")#allenai/longformer-base-4096
# parser.add_argument("--finetune_checkpoint", help="meaning the path of the pre-trained model(for test mode only)", type=str,default="results/checkpoint-4000")#allenai/longformer-base-4096
parser.add_argument("--input_data", help="meaning the path of the input training data", type=str,default="data-final")#data-final-few-shot
parser.add_argument("--num_train_epochs", help="the number of the training epochs", type=int,default=10)
parser.add_argument("--batch_size", help="the batch size of training process", type=int,default=32)


args = parser.parse_args()
print("split way:",args.split)
print("train or test:",args.process)
MODEL_NAME=args.checkpoint#"model/bert-base-uncased";model/allenai/longformer-base-4096"
print("checkpoint path:",MODEL_NAME)
print("num_train_epochs:",args.num_train_epochs)
print("batch_size:",args.batch_size)
INPUT_DATA_PATH=args.input_data
# if args.process=="test":
#     print("checkpoint_path",args.pre_train_model_path)
#     PREDICT_MODEL_NAME=args.pre_train_model_path#"results/checkpoint-4000"

# if args.split=="chunk" or args.split=="sent":
#     MODEL_NAME="model/bert-base-uncased"
# if args.split=="nosplit":
#     MODEL_NAME="model/allenai/longformer-base-4096"

# MODEL_NAME
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)#MODEL_NAME
vocab = tokenizer.get_vocab()

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8"


class EventDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_sents, train_tags = utils.load_dataset(INPUT_DATA_PATH+'/seqlabeling/train.tsv')
dev_sents, dev_tags = utils.load_dataset(INPUT_DATA_PATH+'/seqlabeling/dev.tsv')
test_sents, test_tags = utils.load_dataset(INPUT_DATA_PATH+'/seqlabeling/test.tsv')

if args.split=="chunk" or args.split=="sent":
    train_sents, train_tags = utils.split_dataset(train_sents, train_tags, args.split)
    dev_sents, dev_tags = utils.split_dataset(dev_sents, dev_tags, args.split)
    test_sents, test_tags = utils.split_dataset(test_sents, test_tags, args.split)

if args.split=="chunk":
    MAX_LENGTH=256
if args.split=="sent":
    MAX_LENGTH=64
if args.split=="nosplit":
    MAX_LENGTH=1024
train_encodings = tokenizer(train_sents, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True,max_length=MAX_LENGTH)#
dev_encodings = tokenizer(dev_sents, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True,max_length=MAX_LENGTH)#
test_encodings = tokenizer(test_sents, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True,max_length=MAX_LENGTH)#

with open(INPUT_DATA_PATH+'/seqlabeling/label2id.pkl', 'rb') as fin:
    label2id = pickle.load(fin)

def encode_tags(tags, encodings):
    labels = [[label2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for i, (doc_labels, doc_offset) in enumerate(zip(labels, encodings.offset_mapping)):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)
        tks = tokenizer.convert_ids_to_tokens(encodings['input_ids'][i])

        # set labels whose first offset position is 0 and the second is not 0
        if args.split=="chunk" or args.split=="sent":
            #doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
            length = np.sum(((arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)).astype(np.int32))
            if len(doc_labels) > length:
                doc_labels = doc_labels[:length]
            doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
        if args.split=="nosplit":
            is_first = np.asarray([t.startswith('Ġ') for t in tks])
            length = np.sum(((arr_offset[:,0] == 1) & is_first).astype(np.int32))
            if len(doc_labels) > length:
                doc_labels = doc_labels[:length]
            doc_enc_labels[(arr_offset[:,0] == 1) & is_first] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels

train_labels = encode_tags(train_tags, train_encodings)
dev_labels = encode_tags(dev_tags, dev_encodings)
test_labels = encode_tags(test_tags, test_encodings)

train_encodings.pop("offset_mapping")
dev_encodings.pop("offset_mapping")
test_encodings.pop("offset_mapping")
#print(test_encodings)
print(np.array(test_encodings["input_ids"]).shape)
print(tokenizer.decode(test_encodings["input_ids"][0]))
#print(test_encodings["input_ids"])
#print(tokenizer.decode(test_encodings["input_ids"]))
#print(tokenizer.decode(test_encodings["input_ids"]).shape)

train_dataset = EventDataset(train_encodings, train_labels)
dev_dataset = EventDataset(dev_encodings, dev_labels)
test_dataset = EventDataset(test_encodings, test_labels)


from datasets import load_metric
metric = load_metric("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    predictions = np.reshape(predictions, [-1])
    labels = np.reshape(labels, [-1])
    return metric.compute(predictions=predictions, references=labels)

# if args.split=="chunk":
#     num_train_epochs=10
#     batch_size=32
# if args.split=="sent":
#     num_train_epochs=10
#     batch_size=64
# if args.split=="nosplit":
#     num_train_epochs=6
#     batch_size=2


training_args = TrainingArguments(
    output_dir='./results',                  # output directory
    num_train_epochs=args.num_train_epochs,       # total number of training epochs
    per_device_train_batch_size=args.batch_size,  # batch size per device during training
    per_device_eval_batch_size=args.batch_size,   # batch size for evaluation
    warmup_steps=100,                        # number of warmup steps for learning rate scheduler
    weight_decay=0.01,                       # strength of weight decay
    logging_dir='./logs',                    # directory for storing logs
    logging_steps=10,
    eval_accumulation_steps=6,
    learning_rate=5e-5
)
# if args.process=="train":
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(label2id),ignore_mismatched_sizes=True)
# if args.process=="ft":
    # model = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(label2id),ignore_mismatched_sizes=True)
# if args.process=="train":
#     if args.split=="chunk" or args.split=="sent":
#         model = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(label2id))
#     if args.split=="nosplit":
#         model = LongformerForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(label2id))
# else:
#     if args.split=="chunk" or args.split=="sent":
#         model = BertForTokenClassification.from_pretrained(PREDICT_MODEL_NAME, num_labels=len(label2id))
#     if args.split=="nosplit":
#         model = LongformerForTokenClassification.from_pretrained(PREDICT_MODEL_NAME, num_labels=len(label2id))
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=dev_dataset,             # evaluation dataset
    compute_metrics=compute_metrics,
)

if args.process=="train":
    trainer.train()
    print(trainer.evaluate())

if args.process=="test" or args.process=="train":
    id2label= {v: k for k, v in label2id.items()}
    from sklearn.metrics import f1_score,precision_score,recall_score

    res = trainer.predict(test_dataset)
    # print(res)
    fo=open(INPUT_DATA_PATH+'/seqlabeling/predict_test.tsv',"w")
    # print(res[0].shape)
    # print(np.array(test_encodings["input_ids"]).shape)
    assert len(res[0])==len(test_encodings["input_ids"])
    for preds,words in zip(res[0],test_encodings["input_ids"]):
        # print(words)
        words=tokenizer.convert_ids_to_tokens(words)
        # print(words)
        # print(len(words))
        # print(preds.shape)
        preds=list(np.argmax(preds,axis=-1))
        # print(len(preds))
        for index in range(len(preds)):
            pred=str(id2label[preds[index]])
            word=str(words[index])
            # print(pred,word)
            # if pred!="O":
            #     print(pred)
            #     print(word)
            fo.write(word+"\t"+pred+"\n")
        fo.write("\n")
    fo.close()
            
        
      


    # preds = np.argmax(res[0], axis = -1)
    # real_preds = []
    # real_labels = []
    # other_index = label2id['O']
    # for pred, label in zip(preds, test_labels):
    #     for p, l in zip(pred, label):
    #         if l < len(label2id) and l > 0 and l != other_index:
    #             real_preds.append(p)
    #             real_labels.append(l)
    # '''
    # res = []
    # for pred, label in zip(preds, test_labels):
    #     cr = []
    #     for p, l in zip(pred, label):
    #         if l < len(label2id) and l > 0:
    #             cr.append(p)
    #     res.append(cr)
    # with open('pred_res.pkl', 'wb') as fout:
    #     pickle.dump(res, fout)
    # '''


    # precision = precision_score(real_labels, real_preds, average = 'weighted')
    # recall = recall_score(real_labels, real_preds, average = 'weighted')
    # f = f1_score(real_labels, real_preds, average = 'weighted')
    # print(precision, recall, f)

