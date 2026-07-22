import os
from dotenv import load_dotenv

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

print("Google API Key:")
print(google_api_key)

from google import genai

client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="랭체인과 랭그래프를 비교하여 설명해주세요.",
)

print(response.text)