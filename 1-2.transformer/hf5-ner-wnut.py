from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")

def tokenize_and_align_tags(records):
    # 입력 단어를 토큰으로 분리함. 예를 들어, ChatGPT경우, ['Chat', '##G', '##PT']로 분해
    tokenized_results = tokenizer(records["tokens"], truncation=True, is_split_into_words=True)
    input_tags_list = []

    for i, given_tags in enumerate(records["ner_tags"]):
        word_ids = tokenized_results.word_ids(batch_index=i)
        previous_word_id = None
        input_tags = []

        for wid in word_ids:
            if wid is None:
                input_tags.append(-100)
            elif wid != previous_word_id:
                input_tags.append(given_tags[wid])
            else:
                input_tags.append(-100)
            previous_word_id = wid

        input_tags_list.append(input_tags)

    tokenized_results["labels"] = input_tags_list
    return tokenized_results

from datasets import load_dataset
wnut = load_dataset('wnut_17')
tokenized_wnut = wnut.map(tokenize_and_align_tags, batched=True)
tag_names = wnut["test"].features[f"ner_tags"].feature.names
id2label = dict(enumerate(tag_names))
label2id = dict(zip(id2label.values(), id2label.keys()))

from transformers import AutoModelForTokenClassification
model = AutoModelForTokenClassification.from_pretrained(
    "dslim/bert-base-NER", num_labels=len(id2label), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
)

from transformers import Trainer, TrainingArguments, DataCollatorForTokenClassification
training_args = TrainingArguments(
    output_dir="ner-wnut-model",
)

# 하이퍼파라메터 튜닝용
training_args = TrainingArguments(
    output_dir="ner-wnut-model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)     
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_wnut["train"],
    eval_dataset=tokenized_wnut["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

from transformers import pipeline
model = AutoModelForTokenClassification.from_pretrained("ner-wnut-model/checkpoint-1000")
tokenizer = AutoTokenizer.from_pretrained("wnut_model/checkpoint-1000")

classifier = pipeline("ner", model=model, tokenizer=tokenizer)
out = classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")
print(out)

import evaluate
seqeval = evaluate.load("seqeval")
results = trainer.evaluate()
print(results)
trainer.push_to_hub()