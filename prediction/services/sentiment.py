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
    average_score, _ = analyze_sentiment_details(text_list)
    return average_score


def analyze_sentiment_details(text_list):
    """
    Her haber başlığı için FinBERT etiketi ve etki skorunu üretir.
    Skor aralığı yaklaşık -1 ile +1 arasındadır.
    """
    if isinstance(text_list, str):
        text_list = [text_list]

    text_list = [text for text in text_list if text]
    if not text_list:
        return 0.0, []

    inputs = tokenizer(text_list, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # FinBERT çıktıları: [Positive, Negative, Neutral]
    # Biz (Positive - Negative) yaparak net bir skor elde ediyoruz
    sentiments = predictions.detach().numpy()
    
    # Her haber için (Pozitif - Negatif) farkını alıp ortalamasını bulalım
    details = []
    total_score = 0.0
    for text, s in zip(text_list, sentiments):
        # s[0] = pozitif, s[1] = negatif, s[2] = nötr
        score = float(s[0] - s[1])
        total_score += score

        if score >= 0.15:
            label = "Pozitif"
            sentiment_class = "success"
        elif score <= -0.15:
            label = "Negatif"
            sentiment_class = "danger"
        else:
            label = "Nötr"
            sentiment_class = "secondary"

        details.append({
            "title": text,
            "score": round(score, 4),
            "label": label,
            "class": sentiment_class,
        })

    return total_score / len(text_list), details
