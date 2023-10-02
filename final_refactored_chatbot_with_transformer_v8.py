import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from torch.nn.utils.rnn import pad_sequence

class ChatDataset(Dataset):
    def __init__(self, conversations):
        self.conversations = conversations

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, i):
        return self.conversations[i]

def custom_data_collator(features):
    input_ids = [feature['input_ids'] for feature in features]
    attention_mask = [feature['attention_mask'] for feature in features]
    labels = [feature['labels'] for feature in features]
    
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)  
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

def prepare_dataset(data, model, tokenizer):
    conversations = []
    for article in data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    answer_text = answer['text']

                    input_ids = tokenizer.encode(question, context, add_special_tokens=True, max_length=1024, truncation=True)
                    labels = tokenizer.encode(answer_text, add_special_tokens=False)

                    labels = labels + [-100] * (len(input_ids) - len(labels))

                    if len(input_ids) > 1024:
                        continue

                    attention_mask = [1] * len(input_ids)

                    conversations.append({
                        'input_ids': torch.tensor(input_ids),
                        'attention_mask': torch.tensor(attention_mask),
                        'labels': torch.tensor(labels)
                    })

    dataset = ChatDataset(conversations)

    num_epochs = 2 

    training_args = TrainingArguments(
        output_dir='C:\\Users\\Zacha\\Desktop\\zack\\Final_MSAAI',
        per_device_train_batch_size=8,
        num_train_epochs=num_epochs,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=custom_data_collator
    )

    for epoch in range(num_epochs):
        trainer.train()
        checkpoint_dir = f'C:\\Users\\Zacha\\Desktop\\zack\\Final_MSAAI\\checkpoint-{epoch+1}'
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)

    return trainer

with open("C:\\Users\\Zacha\\Desktop\\zack\\Final_MSAAI\\train-v1.1.json", "r") as f:
    train_data = json.load(f)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

trainer = prepare_dataset(train_data, model, tokenizer)

context = []
while True:
    user_input = input("You: ")
    context.append(user_input)
    input_ids = tokenizer.encode(' '.join(context), return_tensors='pt', max_length=1024, truncation=True)
    output_ids = model.generate(input_ids)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    model_response = output_text.split('.')[-1]
    print(f"Bot: {model_response}")
    context.append(model_response)
