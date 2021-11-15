''' Project description
Full description from canvas:
    Task: Questioning answering
    Data: https://paperswithcode.com/dataset/natural-questions
    Start with yes/no, continue with complex answers

About the data:
    - 42 GB in size - Really F*ing large! Is downloading the best option? or is there a way to fetch from code?

    - The Natural Questions corpus is a question answering dataset containing 307,373 training examples,
    7,830 development examples, and 7,842 test examples.

    - Each example is comprised of a google.com query and a corresponding Wikipedia page.

    - Each Wikipedia page has a passage (or long answer) annotated on the page that answers the question and one or more
    short spans from the annotated passage containing the actual answer.

    - The long and the short answer annotations can however be empty.

    - If they are both empty, then there is no answer on the page at all.

    - If the long answer annotation is non-empty, but the short answer annotation is empty, then the annotated passage
    answers the question but no  explicit short answer could be found.

    - Finally 1% of the documents have a passage annotated with a short answer that is “yes” or “no”,
    instead of a list of short spans.


How data is represented:
    -
'''

# Imports
from transformers import BertTokenizer, BertModel

# What is a tokenizer?
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

model = BertModel.from_pretrained("bert-base-cased")

# Get input data
text = "Replace me by any text you'd like."

encoded_input = tokenizer(text, return_tensors='pt')

output = model(**encoded_input)
