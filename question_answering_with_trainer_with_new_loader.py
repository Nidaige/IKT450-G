### Imports ###
from random import shuffle, random
import numpy as np
from datasets import load_metric, tqdm
import jsonlines as jsonlines
from torch import nn
from torch.autograd.grad_mode import F
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer, pipeline, BertForQuestionAnswering
from transformers.pipelines.base import collate_fn
import torch
import itertools
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
                    if Id == count:  # stop after 20 questions are found, remove when model works
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
        tok_list.append(tok_sample)

        # Shift start and end token labels to window
        shifted_start_label = int(start_label - i*stride)
        shifted_end_label = int(end_label - i*stride)
        if shifted_start_label < 0 or shifted_start_label >= window_size:
            label = torch.Tensor([-1, -1])
        elif shifted_end_label <= 0 or shifted_end_label > window_size:
            label = torch.Tensor([-1,-1])
        else:
            label = torch.Tensor([shifted_start_label, shifted_end_label])

        label_list.append(label)
        i += 1
    return tok_list, label_list

def preprocess_data(path, count):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    torch.multiprocessing.freeze_support()
    NQDataset = NQDataLoader(path,count)
    sliced_dataset = []
    for question in NQDataset:
        x, y = slice_data(question, 512, 128, tokenizer)
        assert (len(x) == len(y)), "You must have as many labels as input samples"
        for i in range(len(y)):
            sliced_dataset.append((x[i], y[i]))
    return sliced_dataset

class dataObject():  # object-version of each data element. Prevent the vars-function in data_collator.py from throwing error
    def __init__(self, x_dict, y_labels):
        self.input_ids = x_dict["input_ids"]
        self.token_type_ids = x_dict["token_type_ids"]
        self.attention_mask = x_dict["attention_mask"]
        self.label = y_labels  # throws error surrounding length of label tensor not being 1
        # self.label = (y_labels[0],y_labels[1])  # throws error surrounding "labels" in batch in forward function
        # self.label_ids = y_labels  # throws error surrounding "labels" in batch in forward function
        # having no label throws error around "too many values to unpack", refers to tuple not containing label.

### Initializing ###
def run():
    ''' Initialization '''
    torch.multiprocessing.freeze_support()

    ''' Loading Dataset'''
    NQDataset = preprocess_data("Data/Project/First20.jsonl", 20)

    ''' Shuffling data'''
    list_of_indices = []
    shuffled_dataset = []
    for l in range(len(NQDataset)):
        list_of_indices.append(l)
    shuffle(list_of_indices)
    for index in list_of_indices:
        shuffled_dataset.append(NQDataset[index])

    ''' Splitting data into training and testing sets'''
    trainset_as_tuples = shuffled_dataset[0:int(0.7*len(shuffled_dataset))]
    testset_as_tuples = shuffled_dataset[int(0.7*len(shuffled_dataset)):]  # only useable when trainer works

    ''' Convert tupled data items into objects for the vars-function in data_collator'''
    trainset = []
    for item in trainset_as_tuples:
        trainset.append(dataObject(item[0],item[1]))
    testset = []
    for item in testset_as_tuples:
        testset.append(dataObject(item[0], item[1]))


    ''' Initialize model params and Trainer class'''
    model = BertForQuestionAnswering.from_pretrained('bert-base-cased')
    #model = AutoModelForQuestionAnswering('bert-base-cased')

    training_args = TrainingArguments('test_trainer')
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=trainset,
        eval_dataset=testset,
        compute_metrics=compute_metrics)

    ''' Run finetuning '''
    trainer.train()
    trainer.evaluate()

    ''' Set up pipeline for question answering, using newly trained model and bert's autotokenizer'''
    nlp = pipeline("question-answering", model=model, tokenizer = BertTokenizer.from_pretrained('bert-base-cased'))

    ''' Get Questions from file '''
    questions = []
    with jsonlines.open("Data/Project/First20.jsonl") as reader:  # open file
        for obj in reader:
            questions.append((obj["question_text"], obj["document_text"], obj["annotations"][-1]["long_answer"]))

    ''' get context, question and answer for test questions'''
    for question in questions:
        query = question[0]
        context = question[1]
        answer_tokens = question[2]
        #tokenized_context = BertTokenizer.from_pretrained('bert-base-cased').tokenize(context)  # messes up answer format
        tokenized_context = context.split(" ")
        pre_answer = tokenized_context[answer_tokens["start_token"]:answer_tokens["end_token"]]
        answer=""
        ''' Print out question, prediction from pipeline and answer from dataset(if any)'''
        for a in pre_answer:
            answer+=a+" "
        print("Question:",query)
        print("Prediction: ",nlp(question=query, context=context)["answer"])
        print("Truth:", answer)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    metric = load_metric('accuracy')
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

if __name__ == '__main__':
    run()