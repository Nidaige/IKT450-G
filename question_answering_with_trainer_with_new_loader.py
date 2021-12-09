### Imports ###
import os
from random import shuffle
import numpy as np
from datasets import load_metric
import jsonlines as jsonlines
from transformers import TrainingArguments, Trainer, BertForQuestionAnswering
import torch
from torch import tensor as tensor
import more_itertools as it
from transformers import BertTokenizer


class NQDataLoader():  # Data loader class for Natural Questions
    def __init__(self,path, count):
        self.yesNoQuestions = []  # Holds all objects with yes-no-answer
        with jsonlines.open(path) as reader:  # open file
            Id = 0
            for obj in reader:
                if obj["annotations"][-1]["yes_no_answer"] == "NONE":  # If the question has a yes/no-answer
                    self.yesNoQuestions.append(obj)  # Add it to the dict
                    Id += 1  # Increment ID
                    print("Reading question: ", Id)
                    if Id == count & count!=0:  # stop after 20 questions are found, remove when model works
                        break

    ''' Train Dataset format
    {
    document_text: str      full html text
    long_answer_candidates: list of dict { start_token: int
                                           top_level: bool
                                           end_token: int }    
    question_text: str (first 4000 questions are 19 or shorter tokens -> combined with answer (2) we'll set aside 26
    annotations: list containing one dict {
                                            yes_no_answer: True/False/NONE
                                            long_answer: dict { start_token: int
                                                                candidate_index: int
                                                                end_token: int }
                                            short_answers: dict{ start_token:int
                                                                 end_token: int }
                                            annotation_id: int }
    document_url: str
    example_id: int
    }
    '''

    def __getitem__(self, idx):
        return self.yesNoQuestions[idx]

    def __len__(self):
        return(len(self.yesNoQuestions))


    # Takes in tokenized data, adds it to the vocabulary. Used for indexing tokens so a tensor can be made
# Each word will be represented by its index in the vocabulary, so that a list of tensors can be instantly translated into words via indexing the list
def mapQuestionsToIndices(tokenized, oldvocab):
    vocab = oldvocab
    for word in tokenized:
        if word not in vocab:
            vocab.append(word)
    return vocab


def slice_data(data, window_size, stride, tokenizer):
    """
    data: {'question': question text,
           'context': context text,
           'labels': (start_token, end_token)
    """
    question = data['question_text']
    context = data['document_text']
    temp_dict = data['annotations'][0]['long_answer']
    start_label, end_label = temp_dict['start_token'], temp_dict['end_token']
    tokenized_context = context.split(' ')
    i=0
    tok_list = []
    label_list = []
    # Slide a window of window_size over the entire context. Stride should be smaller than window size
    # BERT baseline paper for natural questions uses window size 512 and stride 128 if i remembered correctly
    # If stride is smaller than window size you will have overlap in text between the windows but that is nice to have
    # to make sure nothing is missed.
    for window in it.windowed(tokenized_context, n=window_size, step=stride):
        # Filter out None type values added in windowed function
        filtered = list(filter(None, tokenized_context))
        # rejoin tokenized context to context again (BertTokenizer works properly for untokenized sentences)
        window = ' '.join(filtered)
        # join with SEP token between question and context window
        tok_sample = ' <SEP> '.join([question, window])
        # tokenize. truncate and pad if necessary
        tok_sample = tokenizer(tok_sample, padding=True, truncation=True, return_tensors='pt')
        for key in ["input_ids", "token_type_ids", "attention_mask"]:
            tok_sample[key] = tok_sample[key].squeeze()
        tok_list.append(tok_sample)

        # Shift start and end token labels to window
        shifted_start_label = int(start_label - i*stride)
        shifted_end_label = int(end_label - i*stride)
        if shifted_start_label < 0 or shifted_start_label >= window_size:
            label = torch.Tensor([-1, -1]).to(torch.long)
        elif shifted_end_label <= 0 or shifted_end_label > window_size:
            label = torch.Tensor([-1,-1]).to(torch.long)
        else:
            label = torch.Tensor([shifted_start_label, shifted_end_label]).to(torch.long)

        label_list.append(label)
        i += 1

    return tok_list, label_list

def preprocess_data(path, count):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    torch.multiprocessing.freeze_support()
    NQDataset = NQDataLoader(path,count)
    sliced_dataset = []
    j = 0
    for question in NQDataset:
        j+=1
        print("Pre-processing question ",j)
        x, y = slice_data(question, 512, 128, tokenizer)
        assert (len(x) == len(y)), "You must have as many labels as input samples"
        for i in range(len(y)):
            sliced_dataset.append((x[i], y[i]))
    return sliced_dataset


# writes the dataset to a file to skip the pre-processing on subsequent runs
def write_processed_data_to_new_file(source_filepath, destination_filepath, question_count, shuffled, overwrite):
    if not os.path.exists(source_filepath):
        exit(1)
    if os.path.exists(destination_filepath) and not overwrite:
        return
    dataset = preprocess_data(source_filepath, question_count)
    k = 0
    if shuffled:
        list_of_indices = []
        shuffled_dataset = []

        for l in range(len(dataset)):
            list_of_indices.append(l)
        shuffle(list_of_indices)
        for index in list_of_indices:
            shuffled_dataset.append(dataset[index])
        with jsonlines.open(destination_filepath, mode='w') as writer:
            for a in shuffled_dataset:
                k+=1
                print("Writing question ",k)
                writer.write(str(vars(dataObject(a[0],a[1]))))
    else:
        with jsonlines.open(destination_filepath, mode='w') as writer:
            for a in dataset:
                k+=1
                print("Writing question ", k)
                writer.write(str(vars(dataObject(a[0],a[1]))))
    return



class dataObject():  # object-version of each data element. Prevent the vars-function in data_collator.py from throwing error
    def __init__(self, x_dict, y_labels):
        self.input_ids = x_dict["input_ids"]
        self.token_type_ids = x_dict["token_type_ids"]
        self.attention_mask = x_dict["attention_mask"]
        self.start_positions = y_labels[0]
        self.end_positions = y_labels[1]
        #self.label = y_labels  # throws error surrounding length of label tensor not being 1
        # self.label = (y_labels[0],y_labels[1])  # throws error surrounding "labels" in batch in forward function
        # self.label_ids = y_labels  # throws error surrounding "labels" in batch in forward function
        # having no label throws error around "too many values to unpack", refers to tuple not containing label.

class load_dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.dataset = []
        with jsonlines.open(path) as reader:  # open file
            for obj in reader:  # balance numbers of positive and null samples
                item = eval(obj)  # could use same negative samples
                self.dataset.append(item)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]

### Initializing ###
def run():
    ''' Initialization '''
    torch.multiprocessing.freeze_support()



    ''' Loading Dataset'''
    Raw_filepath = "Data/Project/simplified-nq-train.jsonl"
    #Dataset_filepath = "Data/Project/nq-train-fast-read.jsonl"
    Dataset_filepath = "Data/Project/Fast-read.jsonl"
    #write_processed_data_to_new_file(source_filepath=Raw_filepath, destination_filepath=Dataset_filepath,question_count=200, shuffled=True, overwrite = False)
    NQDataset = load_dataset(Dataset_filepath)
    # run dataloader once, read lines by id, load as torch dataset
    ''' Initialize model params and Trainer class'''
    model = BertForQuestionAnswering.from_pretrained('bert-base-cased')
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Cuda available: ",torch.cuda.is_available())
    print("Cuda version: ",torch.version.cuda)
    model.to(device)

    training_args = TrainingArguments('test_trainer', per_device_train_batch_size=1, per_device_eval_batch_size=1)#, prediction_loss_only=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=NQDataset[0:int(0.7*len(NQDataset))],
        eval_dataset=NQDataset[int(0.7*len(NQDataset)):],
        compute_metrics=compute_metrics,
        tokenizer=None
    )

    ''' Run finetuning '''
    trainer.train()
    trainer.evaluate()





def compute_metrics(eval_pred):
    logits, labels = eval_pred
    metric = load_metric('accuracy')
    predictions = np.argmax(logits, axis=-1)
    TP = 0  # True positive - % of prediction that falls inside label
    FP = 0  # False positive - % of prediction that falls outside of label
    FN = 0  # False negative - % of label that is not in prediction
    for prediction_index in range(len(predictions[0])):
        pred_max = max(predictions[0][prediction_index],predictions[1][prediction_index])
        pred_min = min(predictions[0][prediction_index], predictions[1][prediction_index])
        pred_len = pred_max-pred_min
        label_min = labels[0][prediction_index]
        label_max = labels[1][prediction_index]
        label_len = label_max-label_min
        if (label_min == -1 and not pred_min == -1) or (label_max == -1 and not pred_max == -1) or (pred_min == -1 and not label_min == -1) or (pred_max == -1 and not label_max == -1):
            FP += 1
            FN += 1
        else:
            FN += (max(label_min,min(label_max,pred_min))-label_min + label_max-min(label_max,max(pred_max,label_min)))/label_len
            TP += (max(label_min,min(label_max,pred_max))-label_min)/pred_len
            FP += (max(pred_min,label_min)-pred_min + max(pred_max,label_max)-label_max)/pred_len
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    F1 = (2*TP)/(2*TP+FP+FN)
    print("Precision: ",Precision)
    print("Recall: ",Recall)
    print("f1: ",F1)
    exit()
    return metric.compute(predictions=predictions, references=labels)

if __name__ == '__main__':
    run()