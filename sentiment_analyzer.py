import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def calculateDailySentiment(headlines):
    texts = [headline['heading'] for headline in headlines]
    inputs = tokenizer(texts, return_tensors="pt", 
                       truncation=True, padding=True, 
                       max_length=512, return_attention_mask=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        scores = logits.softmax(dim=1)
    averageScore = scores.mean(dim=0).tolist()
    return averageScore

def analyzeAndSaveSentiment(inputFile, outputFile):
    with open(inputFile, 'r') as file:
        data = json.load(file)

    result = {}

    for date, headlines in data.items():
        averageScore = calculateDailySentiment(headlines)
        print(f"{date} > {averageScore}")
        result[date] = averageScore

    with open(outputFile, 'w') as outputFile:
        json.dump(result, outputFile, indent=2)

input_json_file = './data/news/headlines.json'
output_json_file = './data/news/daily_scores.json'

analyzeAndSaveSentiment(input_json_file, output_json_file)