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

    avg_score = scores.mean(dim=0).tolist()
    return avg_score

def analyzeAndSaveSentiment(inputFile, outputFile):
    with open(inputFile, 'r', encoding='utf-8') as file:
        data = json.load(file)

    result = {}

    for date, headlines in data.items():
        avg_score = calculateDailySentiment(headlines)
        print(f"{date} > {avg_score}")
        result[date] = {"negative": avg_score[0], "positive": avg_score[1]}

    with open(outputFile, 'w', encoding='utf-8') as outputFile:
        json.dump(result, outputFile, indent=2)

input_json_file = './data/news/headlines.json'
output_json_file = './data/news/daily_scores.json'

analyzeAndSaveSentiment(input_json_file, output_json_file)