import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import AdamW
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Load the dataset
data = pd.read_csv('/kaggle/input/kaggle-llm-science-exam/train.csv')

# Path to the Roberta tokenizer and model in your Kaggle dataset
ROBERTA_PATH = '/kaggle/input/roberta-base' 

# Load tokenizer and model from the Kaggle dataset path
tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_PATH)
model = RobertaForSequenceClassification.from_pretrained(ROBERTA_PATH, num_labels=5)

# Tokenize the data
input_ids = []
attention_masks = []
labels = []
options = ['A', 'B', 'C', 'D', 'E']

for i, row in data.iterrows():
    text = row['prompt']
    combined_texts = [text + " [SEP] " + row[option] for option in options]
    
    # Tokenize each combined text
    for combined_text in combined_texts:
        inputs = tokenizer.encode_plus(combined_text, add_special_tokens=True, max_length=256, truncation=True, padding='max_length', return_attention_mask=True)
        input_ids.append(inputs['input_ids'])
        attention_masks.append(inputs['attention_mask'])
        labels.append(options.index(row['answer']))

# Convert lists to tensors
input_ids = torch.tensor(input_ids)
attention_masks = torch.tensor(attention_masks)
labels = torch.tensor(labels)

dataset = TensorDataset(input_ids, attention_masks, labels)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

optimizer = AdamW(model.parameters(), lr=2e-5)

scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.85, patience=1)  # Adjust these parameters as needed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0  # reset total loss for each epoch
    for step, batch in enumerate(train_dataloader):
        b_input_ids, b_attention_mask, b_labels = batch
        b_input_ids = b_input_ids.to(device)
        b_attention_mask = b_attention_mask.to(device)
        b_labels = b_labels.to(device)

        model.zero_grad()
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_attention_mask, labels=b_labels)
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_dataloader)
    print("Average training loss for epoch {}: {:.2f}".format(epoch, avg_train_loss))

    # Save model
    torch.save(model.state_dict(), 'roberta_model_weights_epoch_{}.pth'.format(epoch))
    
    # Validation
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    incorrect_samples = []
    for batch in val_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            logits = outputs[1]

            preds = torch.argmax(logits, dim=1)
            eval_accuracy += (preds == b_labels).sum().item()
            incorrect_indices = (preds != b_labels).nonzero(as_tuple=True)[0].cpu().numpy()
            for idx in incorrect_indices:
                incorrect_samples.append((b_input_ids[idx], b_labels[idx], preds[idx]))

    eval_accuracy /= len(val_dataset)
    print('Validation Accuracy for epoch {}: {:.3f}'.format(epoch, eval_accuracy))
    
    # Use eval_accuracy for the scheduler
    scheduler.step(eval_accuracy)
    
    
    
test_data = pd.read_csv('/kaggle/input/kaggle-llm-science-exam/test.csv')

# Tokenize the test data
test_input_ids = []
test_attention_masks = []

for i, row in test_data.iterrows():
    text = row['prompt']
    combined_texts = [text + " [SEP] " + row[option] for option in options]
    
    for combined_text in combined_texts:
        inputs = tokenizer.encode_plus(combined_text, add_special_tokens=True, max_length=256, truncation=True, padding='max_length', return_attention_mask=True)
        test_input_ids.append(inputs['input_ids'])
        test_attention_masks.append(inputs['attention_mask'])

test_input_ids = torch.tensor(test_input_ids)
test_attention_masks = torch.tensor(test_attention_masks)

# Predict using the model
model.eval()
test_dataloader = DataLoader(TensorDataset(test_input_ids, test_attention_masks), batch_size=32)
all_logits = []

for batch in test_dataloader:
    b_input_ids, b_attention_mask = batch
    b_input_ids, b_attention_mask = b_input_ids.to(device), b_attention_mask.to(device)

    with torch.no_grad():
        logits = model(b_input_ids, attention_mask=b_attention_mask)[0]
        all_logits.append(logits)

all_logits = torch.cat(all_logits, dim=0)

# Reshape the logits to get the maximum prediction for each set of 5 options
reshaped_logits = all_logits.view(-1, 5, len(options))
max_preds = torch.argmax(reshaped_logits, dim=2)

# Get the top 3 predictions for each set of 5 options
top_3_preds = torch.topk(max_preds, k=3, dim=1)[1]

# Convert indices to labels
labels_predicted = [[options[idx] for idx in pred] for pred in top_3_preds]

# Create the submission DataFrame
submission = pd.DataFrame({
    'id': range(len(test_data)), 
    'prediction': [' '.join(label_set) for label_set in labels_predicted]
})

# Save the DataFrame as "submission.csv"
submission.to_csv('submission.csv', index=False)

