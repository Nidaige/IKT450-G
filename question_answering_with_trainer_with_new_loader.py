### Imports ###
from random import shuffle, random
import numpy as np
from datasets import load_metric, tqdm
import jsonlines as jsonlines
import numpy
import torch
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer, BertTokenizer, BertModel, AdamW, \
    get_scheduler, pipeline, BertForQuestionAnswering
from transformers.pipelines.base import collate_fn

import new_data_loading





### Initializing ###
def run():
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    torch.multiprocessing.freeze_support()
    NQDataset = new_data_loading.new_data_load("Data/Project/First20.jsonl")


    model = BertForQuestionAnswering.from_pretrained('bert-base-cased')
    #model = BertModel.from_pretrained('bert-base-cased')

    training_args = TrainingArguments('test_trainer')


    # shuffle dataset
    list_of_indices = []
    shuffled_dataset = []
    for l in range(len(NQDataset)):
        list_of_indices.append(l)
    shuffle(list_of_indices)
    print(list_of_indices)
    for index in list_of_indices:
        shuffled_dataset.append(NQDataset[index])
    t = 0
    trainset = shuffled_dataset[0:int(0.7*len(shuffled_dataset))]
    testset = shuffled_dataset[int(0.7*len(shuffled_dataset)):]

    #trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2, collate_fn=collate_fn)
    #testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=True, num_workers=2, collate_fn=collate_fn)


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