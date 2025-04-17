import openai
import os 



from dotenv import load_dotenv
load_dotenv()

# Replace with your actual API key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",  # or "gpt-4" if you have access
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the capital of Japan?"}
    ]
)

print(response.choices[0].message['content'])

