import jsonlines as jsonlines
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, BertTokenizer, AdamW, get_scheduler, pipeline


class NQDataLoader():  # Data loader class for Natural Questions
    def __init__(self):
        self.yesNoQuestions = []  # Holds all objects with yes-no-answer
        with jsonlines.open('Data/Project/simplified-nq-train.jsonl') as reader:  # open file
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




def get20Questions():
    NQDataset = NQDataLoader()

    questions = []
    for i in range(10):
        questions.append([NQDataset[i]["question_text"], NQDataset[i]["document_text"]])
    return questions




'''
NQDataset = NQDataLoader()
for i in range(10):
    data_item = NQDataset.__getitem__(i)
    print("Question: ",data_item["question_text"])
    print("Answer: ",data_item["annotations"][-1]["yes_no_answer"])
    print("-----")'''



def run():
    ''' Used snippets from     # https://huggingface.co/transformers/training.html - fine tuning a pretrained model '''
    
    torch.multiprocessing.freeze_support()
    NQDataset = NQDataLoader()
    data = NQDataset[0]
    print("keys", data.keys())

    # define pre-trained model
    model = AutoModelForQuestionAnswering.from_pretrained("bert-base-cased")
    # define optimizer (AdamW is default for bert)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 3
    num_training_steps = num_epochs * len(NQDataset)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
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
            batch = {k: v.to(device) for k, v in batch.items()} # this line causes crash, figure out = $$$
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)



    nlp = pipeline("question-answering", model=model, tokenizer = BertTokenizer.from_pretrained('bert-base-cased'))
    questions = get20Questions()
    for question in questions:
        query = question[0]
        context = question[1]
        print(query, nlp(question=query, context=context))

if __name__ == '__main__':
    run()