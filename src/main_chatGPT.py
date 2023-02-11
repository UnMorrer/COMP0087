#import src.chatGPT.chatGPT_proxied as chatgpt
import os

if __name__ == "__main__":
    print(f"{os.getcwd()}")
    #main()

    a = 1

def main():
    """
    Runner/orchestration script for chatGPT scraping
    """
    
    chat = chatgpt.Chatbot()