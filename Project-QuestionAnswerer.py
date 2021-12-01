import jsonlines as jsonlines
import numpy
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, BertTokenizer, AdamW, get_scheduler, pipeline


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
def mapQuestionsToIndices(tokenized, oldvocab):
    vocab = oldvocab
    for word in tokenized:
        if word not in vocab:
            vocab.append(word)
    return vocab



def run():
    ''' Used snippets from     # https://huggingface.co/transformers/training.html - fine tuning a pretrained model
    More specifically, the "native pytorch implementation" example near the end.

    For Daniel:
    The attempt here was to convert the question data including the context, question text and answer into a list
    of tensors to that the model could train on them, before using that model in the huggingface pipeline for 
    question answering.
    
    The main issues right now are *how* the answers should be represented as tensors, as the data,
    the simplified train set, does not give a single "right" answer, but rather a dictionary of candidates.
    Each candidate is the start and end token of the proposed answer.
    
    Right now, the "chain" of conversions to tensors is working for the question text and the context, but we're still
    stuck on the answers. Should we use the answer tokens to get the words from the context, and then convert
    those to tensors?
    
    Another issue is that we don't know if the model can accept three tensors, as most other uses earlier and in the
    tutorials online use only two; one for the data and one for the label.

    '''
    torch.multiprocessing.freeze_support()
    NQDataset = NQDataLoader("Data/Project/First20.jsonl")
    list_of_tensors = []
    for question in NQDataset:
        # Initialize tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        # Tokenize context (the wikipedia text), question and answer candidates (List of each element/word etc.)
        tokenized_data = tokenizer.tokenize(question["document_text"])
        tokenized_question = tokenizer.tokenize(question["question_text"])
        '''Stuck on next line, not sure how to proceed to process the answer data'''
        tokenized_answer = tokenizer.tokenize(question["long_answer_candidates"])
        # map the tokens to numbers using a vocabulary function. Update the vocabulary with all the question data.
        vocabulary = mapQuestionsToIndices(tokenized_data,[])
        vocabulary = mapQuestionsToIndices(tokenized_question,vocabulary)
        vocabulary = mapQuestionsToIndices(tokenized_answer,vocabulary) # won't work until we can tokenize the answer
        # convert context, question and answer into indices corresponding to tokens in the vocabulary
        data_as_list_of_indices = []
        question_as_list_of_indices = []
        answer_as_list_of_indices = []
        for context_token in tokenized_data:
            data_as_list_of_indices.append(vocabulary.index(context_token))

        for question_token in tokenized_question:
            question_as_list_of_indices.append(vocabulary.index(question_token))

        for answer_token in tokenized_answer:  # won't work until we can tokenize the answer
            answer_as_list_of_indices.append(vocabulary.index(answer_token))

        # convert from list to numpy array
        data_as_nparray = numpy.ndarray(data_as_list_of_indices)
        question_as_nparray = numpy.ndarray(question_as_list_of_indices)
        answer_as_nparray = numpy.ndarray(answer_as_list_of_indices)
        # convert from numpy array to tensors
        data_as_tensor = torch.tensor(data_as_nparray)
        question_as_tensor = torch.tensor(question_as_nparray)
        answer_as_tensor = torch.tensor(answer_as_nparray)
        # join the tensors together as a single object to feed to the model
        list_of_tensors.append([data_as_tensor,question_as_tensor,answer_as_tensor])
    # define pre-trained model
    model = AutoModelForQuestionAnswering.from_pretrained("bert-base-cased")
    # define optimizer (AdamW is standard for bert)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 3
    num_training_steps = num_epochs * len(NQDataset)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer = optimizer,
        num_warmup_steps = 0,
        num_training_steps = num_training_steps
    )
    # enable GPU if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    # progress bar
    progress_bar = tqdm(range(num_training_steps))

    # start training the model
    model.train()
    for epoch in range(num_epochs):
        for batch in NQDataset:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    # set up pipeline for question answering, using the newly trained model and the auto tokenizer from bert
    nlp = pipeline("question-answering", model=model, tokenizer = BertTokenizer.from_pretrained('bert-base-cased'))
    # for each question, try to find the correct answer.
    '''Once the training works, we'll make this a different set of questions from the training data'''
    for question in NQDataset:
        query = question[0]
        context = question[1]
        print(query, nlp(question=query, context=context))



if __name__ == '__main__':
    #getQuestionsToNewFile()
    run()