from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# FinBERT modelini yüklüyoruz (İnternet hızına göre ilk seferde biraz sürebilir)
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def analyze_sentiment(text_list):
    """
    Haber başlıklarından oluşan bir listeyi alır ve 
    ortalama bir duygu skoru (-1 ile 1 arası) döndürür.
    """
    if not text_list:
        return 0.0

    # Haberleri modelin anlayacağı formata çeviriyoruz
    inputs = tokenizer(text_list, padding=True, truncation=True, return_tensors='pt')
    
    # Tahmin yapıyoruz
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # FinBERT çıktıları: [Positive, Negative, Neutral]
    # Biz (Positive - Negative) yaparak net bir skor elde ediyoruz
    sentiments = predictions.detach().numpy()
    
    # Her haber için (Pozitif - Negatif) farkını alıp ortalamasını bulalım
    total_score = 0
    for s in sentiments:
        # s[0] = pozitif, s[1] = negatif, s[2] = nötr
        score = s[0] - s[1] 
        total_score += score
        
    return total_score / len(text_list)