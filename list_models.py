import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

models = client.models.list()

for m in models:
    name = getattr(m, "name", "")
    # embed destekleyenleri yakalamaya çalışalım
    # (kütüphaneye göre alanlar değişebiliyor, o yüzden güvenli basıyoruz)
    supported = getattr(m, "supported_actions", None) or getattr(m, "supported_methods", None)
    print(name, supported)