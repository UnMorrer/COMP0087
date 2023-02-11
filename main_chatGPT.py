import json

# Custom packages
import src.chatGPT.chatGPT_proxied as chatgpt
import src.save.save_response as save

def main():
    """
    Runner/orchestration script for chatGPT scraping
    """
    with open("config/config.json") as f:
        config = json.loads(f.read())
    chat = chatgpt.Chatbot(config=config)

    # TODO:Select a prompt
    prompt = ""
    prompt_id = 1/2/7/8

    # Get answer
    answer = chat.ask(prompt)
    text = answer["message"]
    conv_id = answer["conversation_id"]

    # Save answer
    save.response(
        string=text,
        file_name=f"{prompt_id}_{conv_id}.txt",
        folder_path="data/responses")

if __name__ == "__main__":
    main()