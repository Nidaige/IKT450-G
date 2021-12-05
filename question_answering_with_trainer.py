### Imports ###
from random import shuffle
import numpy as np
from datasets import load_metric, tqdm
import jsonlines as jsonlines
import numpy
import torch
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer, BertTokenizer, BertModel, AdamW, \
    get_scheduler, pipeline


class NQDataLoader():  # Data loader class for Natural Questions
    def __init__(self,path):
        self.yesNoQuestions = []  # Holds all objects with yes-no-answer
        with jsonlines.open(path) as reader:  # open file
            Id = 0
            for obj in reader:
                if obj["annotations"][-1]["yes_no_answer"] == "NONE":  # If the question has a yes/no-answer
                    self.yesNoQuestions.append(obj)  # Add it to the dict
                    Id += 1  # Increment ID
                    if Id == 100:  # stop after 20 questions are found, remove when model works
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


# Saves the first 20 questions with a long/short answer to a new file for faster testing
def getQuestionsToNewFile():
    questions = NQDataLoader()
    with jsonlines.open('Data/Project/First20.jsonl', mode='w') as writer:
        for a in questions.yesNoQuestions:
            writer.write(a)
    exit()


# Takes in tokenized data, adds it to the vocabulary. Used for indexing tokens so a tensor can be made
# Each word will be represented by its index in the vocabulary, so that a list of tensors can be instantly translated into words via indexing the list
def mapQuestionsToIndices(tokenized, oldvocab):
    vocab = oldvocab
    for word in tokenized:
        if word not in vocab:
            vocab.append(word)
    return vocab


### Initializing ###
def run():
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    torch.multiprocessing.freeze_support()
    NQDataset = NQDataLoader("Data/Project/simplified-nq-train.jsonl")
    list_of_tensors = []
    input_ids = []
    input_id = 0
    c=1
    for question in NQDataset:
        print(c)
        c+=1
        # Tokenize context (the wikipedia text), question and answer candidates (List of each element/word etc.)
        tokenized_data = question["document_text"].split(" ")
        tokenized_question = question["question_text"].split(" ")
        if len(tokenized_question) < 30:  # if less than 480 tokens, populate with 0's
            while len(tokenized_question) < 30:
                tokenized_question.append("0")
        tokenized_answer = [question["annotations"][0]["long_answer"]["start_token"],question["annotations"][0]["long_answer"]["end_token"]]
        start = question["annotations"][0]["long_answer"]["start_token"]
        end = question["annotations"][0]["long_answer"]["end_token"]

        # Bert can only handle 512 tokens
        # For that reason, implementing a sliding window for each question
        # Max question length in first 4000 questions was 19
        # With 2 tokens for answer positions, and some extra space, setting aside 32 tokens for question + answer
        # That means 480 tokens as window size
        # If the answer tokens are not within the window, then predict no answer
        # If one answer token (start/end) is within the window, set edge of window as end token

        vocabulary = mapQuestionsToIndices(tokenized_data,["0"])
        vocabulary = mapQuestionsToIndices(tokenized_question,vocabulary)

        window_size = 480
        step_size = int(window_size/6)  # starting with 6 windows overlapping on the same area
        for window in range(0,len(tokenized_data),step_size):
            input_ids.append(input_id)
            input_id += 1  # print question number
            relative_answer = [start-window,end-window]
            # [start/end][window start/end]
            window_data = tokenized_data[window:window+window_size]  # actual data in window
            if len(window_data) < window_size:  # if less than 480 tokens, populate with 0's
                while len(window_data) < window_size:
                    window_data.append("0")

            # Modify relative answer to fit rules mentioned above
            if relative_answer[0] < 0:  # if answer starts before window
                if relative_answer[1] < 0:  # and answer ends before window
                    relative_answer = [-1,-1]  # set answer to -1,-1
                elif relative_answer[1] < window_size:  # and answer ends inside window:
                    relative_answer = [0,relative_answer[1]]  # set answer to 0,current end
                else:  # and answer ends after window:
                    relative_answer = [0, window_size-1]  # set answer to 0, window_size

            elif relative_answer[0] < window_size:  # answer starts within window
                if relative_answer[1] < window_size:  # answer ends inside window:
                    relative_answer = relative_answer  # do nothing, already correct
                else:  # answer ends after window:
                    relative_answer = [relative_answer[0], window_size-1]  # set answer to 0, window_size

            else:  # answer starts after window
                relative_answer = [-1,-1]

            # convert context, question and answer into indices corresponding to tokens in the vocabulary
            data_as_list_of_indices = []
            question_as_list_of_indices = []
            answer_as_list_of_indices = relative_answer

            for context_token in window_data:
                data_as_list_of_indices.append(vocabulary.index(context_token))

            for question_token in tokenized_question:
                question_as_list_of_indices.append(vocabulary.index(question_token))
            final_list_of_indices = data_as_list_of_indices + question_as_list_of_indices + answer_as_list_of_indices
        # convert from list to numpy array
            final_list_of_indices_as_nparray = numpy.array(data_as_list_of_indices + question_as_list_of_indices)
            # convert from numpy array to tensors
            #list_of_tensors.append((torch.tensor(final_list_of_indices_as_nparray),answer_as_list_of_indices))
            list_of_tensors.append({"outputs":final_list_of_indices_as_nparray, "labels":answer_as_list_of_indices})
        # join the tensors together as a single object to feed to the model
    # define pre-trained model

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    model = BertModel.from_pretrained('bert-base-cased')

    training_args = TrainingArguments('test_trainer')

    metric = load_metric('accuracy')

    # shuffle dataset
    shuffle(input_ids)
    shuffled_dataset = []
    train_data_dict={}
    test_data_dict = {}
    t = 0
    for id in input_ids:
        shuffled_dataset.append(list_of_tensors[id])
        t+=1
        if(t<=7):
            train_data_dict[id]=list_of_tensors[id]
        elif t==11:
            t=0
        else:
            test_data_dict[id]=list_of_tensors[id]


    trainset = shuffled_dataset[0:int(0.7*len(shuffled_dataset))]
    testset = shuffled_dataset[int(0.7*len(shuffled_dataset)):-1]



    '''trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=True, num_workers=2)'''
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=trainset,
        eval_dataset=testset,
        compute_metrics=compute_metrics
)


### Actually running ###

    trainer.train()

    trainer.evaluate()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    metric = load_metric('accuracy')
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

if __name__ == '__main__':
    run()