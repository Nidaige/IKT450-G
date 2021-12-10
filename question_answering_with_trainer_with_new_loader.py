# Imports
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


class NQDataLoader():  # Data loader class that gets each element on the format from the original NQ dataset
    def __init__(self,path, count):
        self.Questions = []  # Holds all objects with yes-no-answer
        with jsonlines.open(path) as reader:  # open jsonl file
            Id = 0  # iterator to print progress
            for obj in reader:
                if obj["annotations"][-1]["yes_no_answer"] == "NONE":  # If the question is not a yes/no question
                    self.Questions.append(obj)  # Add it to the dict
                    Id += 1  # Increment ID
                    print("Reading question: ", Id)
                    if Id == count & count!=0:  # stop when "count" questions have been added
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
        return self.Questions[idx]

    def __len__(self):
        return len(self.Questions)


def slice_data(data, window_size, stride, tokenizer):  # reads data dict from original dataset, returns dict and tensor of label for new data structure
    """
    data: {'question': question text,
           'context': context text,
           'labels': (start_token, end_token)
    """
    question = data['question_text']  # question text
    context = data['document_text']  # wikipedia HTML text
    temp_dict = data['annotations'][0]['long_answer']  # sub-dictionary of start/end tokens of answer
    start_label, end_label = temp_dict['start_token'], temp_dict['end_token']
    tokenized_context = context.split(' ')
    i=0  # iterator variable for progress printing
    tok_list = []
    label_list = []
    # Splitting context into overlapping windows, as was done in the original bert for question answering article
    for window in it.windowed(tokenized_context, n=window_size, step=stride):

        # Filter out None type values added in windowed function
        filtered = list(filter(None, tokenized_context))

        # rejoin tokenized context to context again (BertTokenizer works properly for untokenized sentences)
        window = ' '.join(filtered)

        # join with SEP token between question and context window. Used by BertTokenizer to see sentence start/end
        tok_sample = ' <SEP> '.join([question, window])
        # tokenize. truncate and pad if necessary
        tok_sample = tokenizer(tok_sample, padding=True, truncation=True, return_tensors='pt')  # tokenize
        for key in ["input_ids", "token_type_ids", "attention_mask"]:
            tok_sample[key] = tok_sample[key].squeeze()
        tok_list.append(tok_sample)

        # Adjust answer labels (start/end) to be correct in relation to window position
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


def preprocess_data(path, count):  # turns original dataset into tuple of dict and tensor for further processing
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')  # init tokenizer
    torch.multiprocessing.freeze_support()
    NQDataset = NQDataLoader(path,count)  # load original dataset as list of dict elements
    sliced_dataset = []  # list to hold final list
    j = 0  # iterator variable for progress print
    for question in NQDataset:
        j+=1
        print("Pre-processing question ",j)
        x, y = slice_data(question, 512, 128, tokenizer)  # turn dictionary element into a pair of new dict and label
        assert (len(x) == len(y)), "You must have as many labels as input samples"
        for i in range(len(y)):  # for sample in question:
            sliced_dataset.append((x[i], y[i]))  # add it to the list
    return sliced_dataset


# writes the dataset to a file to skip the pre-processing on subsequent runs
def write_processed_data_to_new_file(source_filepath, destination_filepath, question_count, shuffled, overwrite):
    if not os.path.exists(source_filepath):  # if source file does not exist
        print("Source file not found")
        exit(1)
    if os.path.exists(destination_filepath) and not overwrite:  # if destination file already exists, overwrite is false
        print("Destination file exists, returning as overwrite is turned off")
        return
    dataset = preprocess_data(source_filepath, question_count)
    k = 0
    if shuffled:  # if shuffled is true, shuffle the dataset
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
                writer.write(str(vars(DataObject(a[0], a[1]))))
    else:  # shuffled is false, do not shuffle the dataset
        with jsonlines.open(destination_filepath, mode='w') as writer:
            for a in dataset:
                k+=1
                print("Writing question ", k)
                writer.write(str(vars(DataObject(a[0], a[1]))))
    return


class DataObject():  # class to represent data from original dataset before writing to new file
    def __init__(self, x_dict, y_labels):
        self.input_ids = x_dict["input_ids"]  # input_ids tensor
        self.token_type_ids = x_dict["token_type_ids"]  # token_type_ids tensor
        self.attention_mask = x_dict["attention_mask"]  # attention_mask tensor
        self.start_positions = y_labels[0]  # start position value
        self.end_positions = y_labels[1]  # end position value


class LoadDataset(torch.utils.data.Dataset):  # dataset class, used to hold dataset for trainer
    def __init__(self, path):
        self.dataset = []
        with jsonlines.open(path) as reader:  # open file
            for obj in reader: # for each line
                item = eval(obj)  # read line as dictionary
                self.dataset.append(item)  # add dictionary to dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]

### Initializing ###
def run():
    """Initialization"""
    torch.multiprocessing.freeze_support()

    """Loading Dataset"""
    raw_filepath = "Data/Project/simplified-nq-train.jsonl"  # file to original dataset
    dataset_filepath = "Data/Project/nq-train-fast-read.jsonl"  # file to 200 question data subset
    # dataset_filepath = "Data/Project/Fast-read.jsonl"  # file to 20 question debug dataset
    '''Function call to create new file at Destination_filepath from data at raw_filepath, extracting question_count questions. 
    Shuffles if shuffled is true, overwrites an existing file if overwrite is True.'''
    write_processed_data_to_new_file(source_filepath=raw_filepath, destination_filepath=dataset_filepath,question_count=200, shuffled=True, overwrite = False)
    full_dataset = LoadDataset(dataset_filepath)

    ''' Initialize model params and Trainer class, send model to run on the gpu if possible'''
    model = BertForQuestionAnswering.from_pretrained('bert-base-cased')  # init model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # check if gpu is available
    print("Cuda available: ",torch.cuda.is_available())  # print if gpu is available
    print("Cuda version: ",torch.version.cuda)  # print available cuda version
    model.to(device)  # send model to the gpu
    # Training arguments. Test_trainer is output dir for checkpoints. batch sizes are small as the gpu would fill  up
    training_args = TrainingArguments('test_trainer', per_device_train_batch_size=1, per_device_eval_batch_size=1)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=full_dataset[0:int(0.7*len(full_dataset))],  # train dataset is 70% of the samples
        eval_dataset=full_dataset[int(0.7*len(full_dataset)):],  # test/eval dataset is 30% of the samples
        compute_metrics=compute_metrics,
        tokenizer=None
    )

    ''' Run finetuning '''
    trainer.train()  # training
    trainer.evaluate()  # evaluation


''' Performance metric calculation'''


def compute_metrics(eval_pred):  # originally no manual calculations
    logits, labels = eval_pred  # labels is the true answer positions
    metric = load_metric('accuracy')
    predictions = np.argmax(logits, axis=-1)  # predictions has two lists, containing start indexes and end indexes
    TP = 0  # True positive - % of prediction that falls inside label
    FP = 0  # False positive - % of prediction that falls outside of label
    FN = 0  # False negative - % of label that is not in prediction
    for prediction_index in range(len(predictions[0])):  # for each prediction
        pred_max = max(predictions[0][prediction_index],predictions[1][prediction_index])  # highest predicted index
        pred_min = min(predictions[0][prediction_index], predictions[1][prediction_index])  # lowest predicted index
        pred_len = pred_max-pred_min  # length of prediction interval
        label_min = labels[0][prediction_index]  # start of true answer in label
        label_max = labels[1][prediction_index]  # end of true answer in label
        label_len = label_max-label_min  # length of label
        # If either start or end in label is -1, indicating no answer, but the prediction is not, it's a miss.
        # Same if prediction is -1, but label is not.
        if (label_min == -1 and not pred_min == -1) or (label_max == -1 and not pred_max == -1) or (pred_min == -1 and not label_min == -1) or (pred_max == -1 and not label_max == -1):
            FP += 1  # the entire are covered by the prediction is a false positive
            FN += 1  # the entire true answer is marked as a false negative
        else:  # calculate the proportions of the relevant intervals
            FN += (max(label_min,min(label_max,pred_min))-label_min + label_max-min(label_max,max(pred_max,label_min)))/label_len
            TP += (max(label_min,min(label_max,pred_max))-label_min)/pred_len
            FP += (max(pred_min,label_min)-pred_min + max(pred_max,label_max)-label_max)/pred_len
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1 = (2*TP)/(2*TP+FP+FN)
    print("Precision: ",precision)
    print("Recall: ",recall)
    print("f1: ",f1)
    exit()  # exiting here as the next line causes a crash that was not fixed in time.
    return metric.compute(predictions=predictions, references=labels)


if __name__ == '__main__':
    run()