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


'''
    - If the long answer annotation is non-empty, but the short answer annotation is empty, then the annotated passage
    answers the question but no  explicit short answer could be found.

    - Finally 1% of the documents have a passage annotated with a short answer that is “yes” or “no”,
    instead of a list of short spans.


How data is represented:
    -
'''

NQDataset = NQDataLoader()
for i in range(10):
    data_item = NQDataset.__getitem__(i)
    print("Question: ",data_item["question_text"])
    print("Answer: ",data_item["annotations"][-1]["yes_no_answer"])
    print("-----")
