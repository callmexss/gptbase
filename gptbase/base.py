import json
import logging
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import rich
import tiktoken
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletionChunk, ChatCompletion
from rich.logging import RichHandler


client = OpenAI(max_retries=10)
logger = logging.getLogger(__name__)


logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

def get_message(chat_completion: ChatCompletion) -> str:
    """Extracts the message content from the chat completion."""
    message = chat_completion.choices[0].message.content
    return message


def get_chunks(chat_completion: Stream[ChatCompletionChunk]) -> str:
    """Yields chunks of content from the chat completion."""
    for chunk in chat_completion:
        yield chunk.choices[0].delta.content


def extract_function_call(chat_completion: ChatCompletion) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Extracts the function call name and arguments from the chat completion."""
    chat_completion = response
    if chat_completion.choices:
        message = chat_completion.choices[0].message
        function_call = message.function_call
        if function_call:
            function_name = function_call.name
            function_args = json.loads(function_call.arguments)
            return function_name, function_args
    return None, None


def call_with_chat_completion(chat_completion: Dict[str, Any], func_dic: Dict[str, Any]) -> Any:
    """Calls a function with the chat completion."""
    func_name, func_args = extract_function_call(chat_completion)
    if func_name in func_dic:
        return func_dic[func_name](**func_args)


class FunctionCall:
    def __init__(self, name: str, parameters: Dict[str, Any]):
        self.name = name
        self.parameters = parameters

    def to_dict(self) -> Dict[str, Any]:
        """Converts the FunctionCall instance to a dictionary."""
        return {"name": self.name, "parameters": self.parameters}


class CompletionParameters:
    def __init__(
        self,
        model: str = "gpt-3.5-turbo-0613",
        temperature: float = 1.0,
        top_p: Optional[int] = None,
        max_tokens: Optional[int] = None,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[str] = None,
        n: int = 1,
        stream: bool = False,
        logit_bias: Optional[Dict[str, Any]] = None,
        user: Optional[str] = None,
        function_call: Optional[FunctionCall] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
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
        self.function_call = function_call
        self.functions = functions

    def to_dict(self) -> Dict[str, Any]:
        """Converts the CompletionParameters instance to a dictionary."""
        chat_params = {k: v.to_dict() if isinstance(v, FunctionCall) else v for k, v in vars(self).items() if v is not None}
        return chat_params


class BaseChat:
    def __init__(self, api_key: Optional[str] = None):
        if api_key:
            client.api_key = api_key

    @staticmethod
    def available_model_names() -> List[str]:
        """Returns a list of available model names."""
        return [model.id for model in client.models.list().data]

    def ask(self, prompt: str, system_prompt: str = "", params: CompletionParameters = CompletionParameters()) -> ChatCompletion:
        """Asks a question to the model and returns the response."""
        logger.info(prompt)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        chat_params = params.to_dict()
        chat_params["messages"] = messages

        try:
            chat_completion = client.chat.completions.create(**chat_params)
            return chat_completion
        except Exception as e:
            logger.error(f"Error in creating chat completion: {e}")
            raise

    def build_message(self, role: str, message: str) -> Dict[str, str]:
        """Builds a message dictionary."""
        return {"role": role, "content": message}

    def build_system_message(self, message: str) -> Dict[str, str]:
        """Builds a system message."""
        return self.build_message("system", message)

    def build_user_message(self, message: str) -> Dict[str, str]:
        """Builds a user message."""
        return self.build_message("user", message)

    def build_assistant_message(self, message: str) -> Dict[str, str]:
        """Builds an assistant message."""
        return self.build_message("assistant", message)


class Chat(BaseChat):
    def __init__(self, memory_turns: int = 3, system_prompt: str = ""):
        super().__init__()
        self.system_prompt = system_prompt
        if system_prompt:
            self.q = deque(maxlen=memory_turns + 1)
        else:
            self.q = deque(maxlen=memory_turns)

    def ask(self, messages: deque, system_prompt: str = "", params: CompletionParameters = CompletionParameters()) -> Dict[str, Any]:
        """Asks a question to the model and returns the response."""
        logger.info(messages)
        messages = [
            {"role": "system", "content": system_prompt},
            *messages,
        ]

        chat_params = params.to_dict()
        chat_params["messages"] = messages

        chat_completion = client.chat.completions.create(**chat_params)

        return chat_completion

    def chat(self, message: str, params: CompletionParameters = CompletionParameters()) -> str:
        """Chats with the model and returns the response."""
        self.q.append(self.build_user_message(message))
        chat_completion = self.ask(self.q, self.system_prompt, params)

        if not params.stream:
            message = get_message

            message = get_message(chat_completion)
            rich.print(message)
        else:
            li = []
            for chunk in get_chunks(chat_completion):
                if not chunk:
                    continue

                rich.print(chunk, end="")
                li.append(chunk)
            print()
            message = "".join(li)

        self.q.append(self.build_assistant_message(message))
        return message


# Constants for token counts and model encodings
TOKENS_PER_MESSAGE = {"gpt-3.5-turbo-0301": 4, "gpt-4-0314": 3}
TOKENS_PER_NAME = {"gpt-3.5-turbo-0301": -1, "gpt-4-0314": 1}
DEFAULT_MODEL = "cl100k_base"
WARNINGS = {
    "gpt-3.5-turbo": "Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.",
    "gpt-4": "Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.",
}


def get_encoding(model):
    """Get the encoding for a model, with a fallback to the default model if not found."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        print(f"Warning: model {model} not found. Using {DEFAULT_MODEL} encoding.")
        return tiktoken.get_encoding(DEFAULT_MODEL)


def count_tokens(text: str, model="gpt-3.5-turbo"):
    """
    >>> count_tokens('hello world')
    2
    >>> count_tokens('hello world!')
    3
    """
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def count_tokens_in_message(message, encoding, model):
    """Count the tokens in a single message."""
    tokens = TOKENS_PER_MESSAGE.get(model, 0)
    for key, value in message.items():
        tokens += len(encoding.encode(value))
        if key == "name":
            tokens += TOKENS_PER_NAME.get(model, 0)
    return tokens


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


if __name__ == "__main__":
    cm = CompletionParameters()

    assistant = BaseChat()
    response = assistant.ask("What is the weather today?")
    message = response.choices[0].message.content
    print(message)

    cm.stream = True
    response = assistant.ask("How to learn python?", params=cm)
    li = []
    for chunk in get_chunks(response):
        if not chunk:
            continue
        li.append(chunk)
        print(chunk, end="")
    print()

    s = ''.join(li)
    msg = assistant.build_assistant_message(s)
    num_tokens_from_messages([msg])

    assistant = Chat()
    cm.stream = True
    _ = assistant.chat("I want to learn python", cm)
    _ = assistant.chat("Thanks for your advice, do you have any other suggestions?", cm)

    # Define the function
    get_weather_function = FunctionCall("get_current_weather", {"location": "Boston"})

    # Define the functions that the model can call
    functions = [
        {
            "name": "get_current_weather",
            "description": "Gets the current weather for a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get the weather for"
                    }
                },
                "required": ["location"]
            }
        }
    ]

    # Define the parameters for the chat completion
    params = CompletionParameters(
        model="gpt-3.5-turbo-0613",
        function_call=get_weather_function,
        functions=functions,
        # Add other parameters as needed
    )

    # Create an instance of the Assistant class
    assistant = BaseChat()

    # Call the function
    response = assistant.ask("What's the weather like in Boston?", params=params)

    # Print the response
    print(response)
    extract_function_call(response)

    def get_current_weather(location):
        """fake get current weather function."""
        print(f"Today's fake weather for {location} is _°")

    func_dic = {get_current_weather.__name__: get_current_weather}
    func_dic
    call_with_chat_completion(response, func_dic)
