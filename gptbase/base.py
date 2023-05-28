import logging

import openai
import rich


logging.basicConfig(level=logging.INFO)


class Assistant:
    def __init__(self, api_key=None):
        if api_key:
            openai.api_key = api_key

    @staticmethod
    def available_model_names():
        return [model["id"] for model in openai.Model.list()["data"]]

    @staticmethod
    def ask(prompt: str, system_prompt="", model="gpt-3.5-turbo"):
        logging.info(prompt)
        chatgpt = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        message = chatgpt.choices[0]["message"]["content"]
        rich.print(f"[red]{message}[/red]")
        return message, chatgpt

    @staticmethod
    def ask_stream(prompt: str, system_prompt=""):
        logging.info(prompt)
        reponse = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )
        return reponse


if __name__ == "__main__":
    import time

    s = time.time()
    for chunk in Assistant.ask_stream("hello"):
        print(f'{chunk["choices"][0]["delta"].get("content", ""):>12}', end="--------")
        print(time.time() - s)

    s = time.time()
    Assistant.ask("hello")
    print(time.time() - s)
