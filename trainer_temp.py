### Imports ###
from transformers import TrainingArguments, Trainer, BertTokenizer, BertModel
import numpy as np
from datasets import load_metric


### Initializing ###

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

model = BertModel.from_pretrained('bert-base-cased')

training_args = TrainingArguments('test_trainer')

metric = load_metric('accuracy')


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=PUT_DATASET_HERE,  # TODO: Put datasets here
        eval_dataset=PUT_DATASET_HERE,
        compute_metrics=compute_metrics
)


### Actually running ###

trainer.train()

trainer.evaluate()
