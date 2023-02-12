import json
import numpy as np
import time
import asyncio

# Custom packages
import src.chatGPT.chatGPT as chatgpt
import src.save.save_response as save

scrape_per_prompt = 3000

async def main():
    """
    Runner/orchestration script for chatGPT scraping
    """
    with open("config/config.json") as f:
        config = json.loads(f.read())
    chat = chatgpt.Chatbot(config["email"], config["password"])

    prompts = {}
    for prompt_num in ["q1", "q2", "q7", "q8"]:
        with open(f"data/prompts/{prompt_num}.txt") as f:
            prompts[prompt_num] = f.read()

    # Loop over messages
    for message_num in range(55, scrape_per_prompt):
        for prompt_num in prompts.keys():
            prompt = prompts[prompt_num]
            # Get answer
            answer = chat.ask(prompt)
            
            # Print answer
            text = ""
            async for line in answer:
                text += line["choices"][0]["text"].replace("<|im_end|>", "")

            # Save answer
            save.response(
                string=text,
                file_name=f"{prompt_num}_{message_num}.txt",
                folder_path="data/responses")
            
            # Wait between queries
            wait_time = np.random.normal(loc=16.7, scale=3)
            time.sleep(wait_time)

if __name__ == "__main__":
    asyncio.run(main())