import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found")

client = genai.Client(api_key=api_key)

chat = client.chats.create(
    model="gemini-3.5-flash"
)

print("Gemini Chatbot")
print("Type 'exit' to quit.\n")

while True:

    user_message = input("You: ")

    if user_message.lower() == "exit":
        print("Goodbye!")
        break

    response = chat.send_message(
        message=user_message
    )

    print("\nGemini:", response.text)
    print()