import torch
import jsonlines as jsonlines
import itertools
import more_itertools as it
from transformers import BertTokenizer
import numpy as np

class NQDataLoader():  # Data loader class for Natural Questions
    def __init__(self,path):
        self.yesNoQuestions = []  # Holds all objects with yes-no-answer
        with jsonlines.open(path) as reader:  # open file
            Id = 0
            for obj in reader:
                if obj["annotations"][-1]["yes_no_answer"] == "NONE":  # If the question has a yes/no-answer
                    self.yesNoQuestions.append(obj)  # Add it to the dict
                    Id += 1  # Increment ID
                    if Id == 20:  # stop after 20 questions are found, remove when model works
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

def new_data_load(path):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    torch.multiprocessing.freeze_support()
    NQDataset = NQDataLoader(path)

    sliced_dataset = []
    for question in NQDataset:
        x, y = slice_data(question, 512, 128, tokenizer)
        assert (len(x) == len(y)), "You must have as many labels as input samples"
        for i in range(len(y)):
            sliced_dataset.append((x[i], y[i]))
    return sliced_dataset
# sliced_dataset is now a list containing tuples (x, y) where x is a dictionary and y is a torch.tensor of the form [start, end]
# It will contain all window slices from all questions and context given in the datafile
# You can generate batches and your train and test set from from this list and so on.
# Keep in mind: With the window slicing you have a lot of negative examples (samples with no answer [-1.-1]) which creates an imbalance between the classes. You might have to balance this. I think this is also mentioned in the bert baseline paper for natural questions.
# Also, this is not particularly fast code
