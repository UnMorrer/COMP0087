import src.chatGPT.chatGPT_proxied as chatgpt
import os
import json

def main():
    """
    Runner/orchestration script for chatGPT scraping
    """
    with open("config/config.json") as f:
        config = json.loads(f.read())
    chat = chatgpt.Chatbot(config=config)

    prompt = "Hello, how are you?"
    answer = chat.ask(prompt)
    text = answer["message"]
    pass

if __name__ == "__main__":
    main()