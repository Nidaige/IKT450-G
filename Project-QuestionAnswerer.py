import jsonlines as jsonlines
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

class NQDataLoader():  # Data loader class for Natural Questions
    def __init__(self):
        self.yesNoQuestions = []  # Holds all objects with yes-no-answer
        with jsonlines.open('Data/Project/v1.0-simplified_nq-dev-all.jsonl') as reader:  # open file
            Id = 0
            for obj in reader:
                if obj["annotations"][-1]["yes_no_answer"] != "NONE":  # If the question has a yes/no-answer
                    self.yesNoQuestions.append(obj)  # Add it to the dict
                    Id += 1  # Increment ID
                    if Id == 20:  # stop after 20 questions are found, remove when model works
                        break

    def __getitem__(self, idx):
        return self.yesNoQuestions[idx]

    def __len__(self):
        return(len(self.yesNoQuestions))

    def getValuesForKeys(self):
        vals = []
        for a in self.yesNoQuestions:
            value = []
            for b in a.keys():
                print(b)
                print(a[b])
            exit()


def get20Questions():
    NQDataset = NQDataLoader()
    print(NQDataset[0].keys())
    questions = []
    for i in range(10):
        questions.append([NQDataset[i]["question_text"], NQDataset[i]["document_html"]])
    return questions




'''
NQDataset = NQDataLoader()
for i in range(10):
    data_item = NQDataset.__getitem__(i)
    print("Question: ",data_item["question_text"])
    print("Answer: ",data_item["annotations"][-1]["yes_no_answer"])
    print("-----")'''



def run():
    torch.multiprocessing.freeze_support()
    NQDataset = NQDataLoader()
    NQDatasetValues = NQDataset.getValuesForKeys()
    for NQDatasetValue in NQDatasetValues:
        print("-----")
        for a in NQDatasetValue:
            print(NQDatasetValue,a)
    exit()

    from transformers import pipeline

    nlp = pipeline("question-answering", model=AutoModelForQuestionAnswering.from_pretrained("bert-based-cased"), tokenizer=AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad"))
    questions = get20Questions()
    for question in questions:
        query = question[0]
        context = question[1]
        print(query, nlp(question=query, context=context))

if __name__ == '__main__':
    run()