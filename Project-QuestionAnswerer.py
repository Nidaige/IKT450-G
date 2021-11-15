import jsonlines as jsonlines


class NQDataLoader():  # Data loader class for Natural Questions
    def __init__(self):
        self.yesNoQuestions = {}  # Holds all objects with yes-no-answer
        with jsonlines.open('Data/Project/v1.0-simplified_nq-dev-all.jsonl') as reader:  # open file
            Id = 0
            for obj in reader:
                if obj["annotations"][-1]["yes_no_answer"] != "NONE":  # If the question has a yes/no-answer
                    self.yesNoQuestions[Id] = (obj)  # Add it to the dict
                    Id += 1  # Increment ID
                    if Id == 20:  # stop after 20 questions are found, remove when model works
                        break

    def __getitem__(self, idx):
        return self.yesNoQuestions[idx]

    def __len__(self):
        return(len(self.yesNoQuestions))



NQDataset = NQDataLoader()
print(NQDataset.__getitem__(11)["annotations"][-1]["yes_no_answer"])