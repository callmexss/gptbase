import logging
from collections import deque

import openai
import rich
from rich.logging import RichHandler


# Setup logger
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger("rich")


def get_message(chat_completion):
    message = chat_completion.choices[0].message.content
    return message


def get_chunks(chat_completion):
    for chunk in chat_completion:
        yield chunk["choices"][0]["delta"].get("content", "")


class CompletionParameters:
    def __init__(
        self,
        model="gpt-3.5-turbo",
        temperature=1.0,
        top_p: int = None,
        max_tokens=None,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=None,
        n=1,
        stream=False,
        logit_bias=None,
        user=None,
    ):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop
        self.n = n
        self.stream = stream
        self.logit_bias = logit_bias
        self.user = user

    def to_dict(self):
        # create a dictionary with only non-None parameters
        chat_params = {k: v for k, v in vars(self).items() if v is not None}
        return chat_params


class Assistant:
    def __init__(self, api_key=None):
        if api_key:
            openai.api_key = api_key

    @staticmethod
    def available_model_names():
        return [model["id"] for model in openai.Model.list()["data"]]

    def ask(self, prompt: str, system_prompt="", params=CompletionParameters()):
        logger.info(prompt)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        chat_params = params.to_dict()
        chat_params["messages"] = messages

        chat_completion = openai.ChatCompletion.create(**chat_params)

        return chat_completion

    def build_message(self, role, message):
        return {"role": role, "content": message}

    def build_system_message(self, message):
        return self.build_message("system", message)

    def build_user_message(self, message):
        return self.build_message("user", message)

    def build_assistant_message(self, message):
        return self.build_message("assistant", message)


class ChatAssistant(Assistant):
    def __init__(self, memory_turns=3, system_prompt=""):
        super().__init__()
        self.system_prompt = system_prompt
        if system_prompt:
            self.q = deque(maxlen=memory_turns + 1)
        else:
            self.q = deque(maxlen=memory_turns)

    def ask(self, messages: deque, system_prompt="", params=CompletionParameters()):
        logger.info(messages)
        messages = [
            {"role": "system", "content": system_prompt},
            *messages,
        ]

        chat_params = params.to_dict()
        chat_params["messages"] = messages

        chat_completion = openai.ChatCompletion.create(**chat_params)

        return chat_completion

    def chat(self, message: str, params=CompletionParameters()):
        self.q.append(self.build_user_message(message))
        chat_completion = self.ask(self.q, self.system_prompt, params)

        if not params.stream:
            message = get_message(chat_completion)
            rich.print(message)
        else:
            li = []
            for chunk in get_chunks(chat_completion):
                rich.print(chunk, end="")
                li.append(chunk)
            print()
            message = "".join(li)

        self.q.append(self.build_assistant_message(message))
        return message


if __name__ == "__main__":
    cm = CompletionParameters()

    assistant = Assistant()
    response = assistant.ask("What is the weather today?")
    message = response.choices[0].message.content
    print(message)

    assistant = ChatAssistant()
    cm.stream = True
    _ = assistant.chat("I want to learn python", cm)
    _ = assistant.chat("Thanks for your advice, do you have any other suggestions?", cm)
