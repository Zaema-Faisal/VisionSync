import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

if not API_KEY:
    raise RuntimeError("Gemini API key not found in .env")

client = genai.Client(api_key=API_KEY)

response = client.models.generate_content(
    model="gemini-3.5-flashn-lite",
    contents="You are VisionSync. Introduce yourself briefly."
)

print(response.text)