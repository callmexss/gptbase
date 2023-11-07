import json
import logging
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Callable

import rich
import tiktoken
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletionChunk, ChatCompletion
from openai.types.audio import Transcription
from rich.logging import RichHandler

import const


client = OpenAI(max_retries=10)
logger = logging.getLogger(__name__)


logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)


def openai_tts(
    input: str,
    model: str = const.TTS_1,
    voice: str = const.AIVoice.ALLOY,
    response_format: str = const.TTSAudioType.MP3,
    speed: float = 1.0,
):
    client = OpenAI()
    return client.audio.speech.create(
        input=input,
        model=model,
        voice=voice,
        response_format=response_format,
        speed=speed,
    )


def openai_whisper_common(
    audio_file_path: str,
    model: str,
    api_function: Callable[..., Any],
    **kwargs
):
    with open(audio_file_path, "rb") as audio_file:
        result = api_function(
            model=model,
            file=audio_file,
            **kwargs
        )
    return result


def openai_whisper(
    audio_file_path: str,
    model: str = const.WHISPER_1,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    response_format: str = const.WhisperResponseType.JSON,
    temperature: Optional[float] = None
):
    return openai_whisper_common(
        audio_file_path,
        model,
        client.audio.transcriptions.create,
        language=language,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature
    )


def openai_whisper_translation(
    audio_file_path: str,
    model: str = const.WHISPER_1,
    prompt: Optional[str] = None,
    response_format: str = const.WhisperResponseType.JSON,
    temperature: Optional[float] = None
):
    return openai_whisper_common(
        audio_file_path,
        model,
        client.audio.translations.create,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature
    )


class TranscriptionConverter:
    def __init__(self, transcription: Transcription):
        self.transcription = transcription
        self.segments = transcription.segments

    def to_text(self) -> str:
        return '\n'.join(segment['text'] for segment in self.segments)

    def to_json(self) -> str:
        return json.dumps([segment['text'] for segment in self.segments])

    def to_srt(self) -> str:
        return self._convert_to_subtitle_format(is_srt=True)

    def to_vtt(self) -> str:
        return 'WEBVTT\n\n' + self._convert_to_subtitle_format(is_srt=False)

    def _convert_to_subtitle_format(self, is_srt: bool) -> str:
        subtitle_output = []
        for i, segment in enumerate(self.segments):
            start_time = segment['start']
            end_time = segment['end']
            start_timestamp = self._format_timestamp(start_time, is_srt)
            end_timestamp = self._format_timestamp(end_time, is_srt)
            subtitle_output.append(f'{i + 1}\n{start_timestamp} --> {end_timestamp}\n{segment["text"]}\n')
        return '\n'.join(subtitle_output)

    @staticmethod
    def _format_timestamp(time: float, is_srt: bool) -> str:
        hours, remainder = divmod(int(time), 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int((time % 1) * 1000)
        if is_srt:
            return f'{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}'
        else:
            return f'{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}'

    def convert(self, output_type: str) -> str:
        if output_type == const.WhisperResponseType.TEXT:
            return self.to_text()
        elif output_type == const.WhisperResponseType.JSON:
            return self.to_json()
        elif output_type == const.WhisperResponseType.SRT:
            return self.to_srt()
        elif output_type == const.WhisperResponseType.VTT:
            return self.to_vtt()
        else:
            return json.dumps(self.transcription)


class ImageParameters:
    def __init__(
        self,
        prompt: str = "",
        n: int = 1,
        size: str = const.ImgSize._1024x1024,
        model: str = const.DALLE_2,
        quality: Optional[str] = None,  # Only for DALL-E 3
        style: Optional[str] = None,    # Only for DALL-E 3
        response_format: str = const.DallEResponseType.URL,
        user: Optional[str] = None,
    ):
        self.prompt = prompt
        self.n = n
        self.size = size
        self.model = model
        self.quality = quality
        self.style = style
        self.response_format = response_format
        self.user = user

    def to_dict(self) -> dict:
        params = {
            "n": self.n,
            "size": self.size,
            "model": self.model,
            "response_format": self.response_format,
        }
        if self.prompt:
            params["prompt"] = self.prompt
        if self.quality:  # Only add if DALL-E 3 and parameter is provided
            params["quality"] = self.quality
        if self.style:    # Only add if DALL-E 3 and parameter is provided
            params["style"] = self.style
        if self.user:
            params["user"] = self.user
        return params


class DallE:
    def __init__(self, api_key: str = None):
        if api_key:
            client.api_key = api_key

    def generate_image(self, image_params: ImageParameters) -> List[dict]:
        payload = image_params.to_dict()
        return client.images.generate(**payload)

    def edit_image(self, image: bytes, mask: Optional[bytes], image_params: ImageParameters) -> List[dict]:
        payload = image_params.to_dict()
        payload["image"] = image
        if mask:
            payload["mask"] = mask
        return client.images.edit(**payload)

    def create_image_variation(self, image: bytes, image_params: ImageParameters) -> List[dict]:
        payload = image_params.to_dict()
        payload["image"] = image
        return client.images.create_variation(**payload)


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


class ToolFunction:
    def __init__(self, name: str, parameters: Dict[str, Any], description: Optional[str] = None):
        self.type = "function"
        self.function = {
            "name": name,
            "parameters": parameters,
            "description": description
        }

    def to_dict(self) -> Dict[str, Any]:
        """Converts the ToolFunction instance to a dictionary."""
        return {
            "type": self.type,
            "function": self.function
        }

class ToolChoice:
    def __init__(self, name: Optional[str] = None):
        if name:
            self.tool_choice = {
                "type": "function",
                "function": {
                    "name": name
                }
            }
        else:
            self.tool_choice = "auto"

    def to_dict(self) -> Dict[str, Any]:
        """Converts the ToolChoice instance to a dictionary."""
        return self.tool_choice


class CompletionParameters:
    def __init__(
        self,
        model: str = const.GPT_35_TURBO,
        temperature: float = 1.0,
        top_p: Optional[int] = None,
        max_tokens: Optional[int] = None,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None,
        n: int = 1,
        stream: bool = False,
        logit_bias: Optional[Dict[str, float]] = None,
        user: Optional[str] = None,
        function_call: Optional[FunctionCall] = None,  # Deprecated: function_call
        functions: Optional[List[Dict[str, Any]]] = None, # Deprecated: functions
        tools: Optional[List[ToolFunction]] = None,
        tool_choice: Optional[ToolChoice] = None,
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
        self.tools = [tool.to_dict() for tool in tools] if tools else None
        self.tool_choice = tool_choice.to_dict() if tool_choice else None

    def to_dict(self) -> Dict[str, Any]:
        """Converts the CompletionParameters instance to a dictionary."""
        chat_params = {
            k: v.to_dict() if hasattr(v, "to_dict") else v 
            for k, v in vars(self).items() 
            if v is not None
        }
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


def get_encoding(model):
    """Get the encoding for a model, with a fallback to the default model if not found."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        print(f"Warning: model {model} not found. Using {const.DEFAULT_MODEL} encoding.")
        return tiktoken.get_encoding(const.DEFAULT_MODEL)


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
    tokens = const.TOKENS_PER_MESSAGE.get(model, 0)
    for key, value in message.items():
        tokens += len(encoding.encode(value))
        if key == "name":
            tokens += const.TOKENS_PER_NAME.get(model, 0)
    return tokens


def num_tokens_from_messages(messages, model=const.GPT_35_TURBO_0613):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model in const.GPT_MODEL_LIST:
        tokens_per_message = 3
        tokens_per_name = 1
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
        model=const.GPT_35_TURBO_0613,
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

    s = 'Python 是一门非常有趣的编程语言。'
    response = openai_tts(s, speed=1.2)
    response.stream_to_file('./test.mp3')
    res = openai_whisper('./test.mp3', response_format=const.WhisperResponseType.VERBOSE_JSON)
    rich.print(res)
    converter = TranscriptionConverter(res)
    rich.print(converter.to_srt())
    rich.print(converter.to_vtt())
    openai_whisper_translation('./test.mp3', response_format=const.WhisperResponseType.VTT)

    image_client = DallE()
    # Image generation with DALL-E 3
    image_params = ImageParameters(
        prompt="A cute baby sea otter",
        n=1,
        size=const.ImgSize._1024x1024,
        model=const.DALLE_3,
        quality=const.ImgQuality.STANDARD,
        style=const.ImgStyle.VIVID,
    )
    generated_images = image_client.generate_image(image_params)

    # Image editing with DALL-E 2 (since editing is only supported with DALL-E 2 as per provided doc)
    image_params_dalle2 = ImageParameters(
        prompt="a love mask used for edit mask",
        n=2,
        size=const.ImgSize._256x256,
        model=const.DALLE_2,
    )
    generated_images = image_client.generate_image(image_params_dalle2)
    edited_images = image_client.edit_image(
        image=open("cloud.png", "rb").read(),
        mask=open("cloud.png", "rb").read(),
        image_params=image_params_dalle2
    )

    # Creating image variations with DALL-E 2
    image_variation_params = ImageParameters(
        n=2,
        size=const.ImgSize._256x256,
        model=const.DALLE_2,
    )
    image_variations = image_client.create_image_variation(
        image=open("mask.png", "rb").read(),
        image_params=image_variation_params
    )

    # Process the response as needed
    print(generated_images)
    print(edited_images)
    print(image_variations)
