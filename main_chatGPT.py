import json
import numpy as np
import time

# Custom packages
import src.chatGPT.chatGPT_proxied as chatgpt
import src.save.save_response as save

scrape_per_prompt = 2500

def main():
    """
    Runner/orchestration script for chatGPT scraping
    """
    with open("config/config.json") as f:
        config = json.loads(f.read())
    chat = chatgpt.Chatbot(config=config)
    chat.ask("Hello!")

    prompts = {}
    for prompt_num in ["q1", "q2", "q7", "q8"]:
        with open(f"data/prompts/{prompt_num}.txt") as f:
            prompts[prompt_num] = f.read()

    # Loop over messages
    for prompt_num in prompts.keys():
        prompt = prompts[prompt_num]
        # Wait between queries
        wait_time = np.random.normal(loc=70, scale=5)
        time.sleep(wait_time)

        # Get 1st answer
        answer = chat.ask(prompt)

        # Save answer
        save.response(
            string=text,
            file_name=f"{prompt_num}_0.txt",
            folder_path="data/responses")

        for message_num in range(1, scrape_per_prompt):

            # Wait between queries
            wait_time = np.random.normal(loc=70, scale=5)
            time.sleep(wait_time)

            # Get answer
            answer = chat.ask(prompt)
            text = answer["message"]

            # Save answer
            save.response(
                string=text,
                file_name=f"{prompt_num}_{message_num}.txt",
                folder_path="data/responses")

if __name__ == "__main__":
    main()