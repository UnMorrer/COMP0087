import json
import numpy as np
import time

# Custom packages
import src.chatGPT.chatGPT as chatgpt
import src.save.save_response as save

init = 615
offset = 425

def main():
    """
    Runner/orchestration script for chatGPT scraping
    """
    with open("config/config.json") as f:
        config = json.loads(f.read())
    chat = chatgpt.Chatbot(config)

    prompts = {}
    for prompt_num in ["q1", "q2", "q7", "q8"]:
        with open(f"data/prompts/{prompt_num}.txt") as f:
            prompts[prompt_num] = f.read()

    # Loop over messages
    for message_num in range(init, init+offset):
        for prompt_num in prompts.keys():
            prompt = prompts[prompt_num]

            # Get answer
            for data in chat.ask(prompt):
                response = data["message"]

            # Save answer
            save.response(
                string=response,
                file_name=f"{prompt_num}_{message_num}.txt",
                folder_path="data/responses")
            
            # Wait between queries
            wait_time = np.random.normal(loc=16.7, scale=3)
            time.sleep(wait_time)

if __name__ == "__main__":
    main()