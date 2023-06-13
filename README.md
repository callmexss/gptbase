# 🚀 GPTBase

Welcome to GPTBase! This Python library provides a straightforward and user-friendly interface to leverage OpenAI's powerful GPT-3.5-Turbo and GPT-4 models for text generation. 📚🔍

## ✨ Features

- 🎛️ Configurable parameters for chat completion.
- 💬 Support for both individual prompts and chat conversations.
- ⚡ Stream output option for faster response times.
- 📝 Built-in logging and rich printing for enhanced visualization.
- 🔄 Customizable model choice, including GPT-3.5-Turbo, GPT-4, and other variants.
- 📈 Easy model parameter adjustment.
- 🤖 Support for function calls in chat completions.

## 📦 Installation

You can easily install the GPTBase library from PyPI with a single command line:

```bash
pip install gptbase
```

Once installed, import it in your Python script:

```python
from gptbase.basev2 import ChatAssistant, CompletionParameters, FunctionCall
```

## 📚 Usage

The library includes two powerful classes: `Assistant` for individual prompts and `ChatAssistant` for engaging, multi-turn chat conversations.

Here's a snippet showcasing a simple usage of the `ChatAssistant` class:

```python
from gptbase.basev2 import ChatAssistant, CompletionParameters

cm = CompletionParameters(stream=True)
assistant = ChatAssistant(memory_turns=3)
message = assistant.chat("I want to learn Python", cm)
```

And here's an example of how to use function calls:

```python
from gptbase.basev2 import Assistant, CompletionParameters, FunctionCall

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
assistant = Assistant()

# Call the function
response = assistant.ask("What's the weather like in Boston?", params=params)

# Print the response
print(response)
extract_function_call(response)

def get_current_weather(location):
    """fake get current weather function."""
    print(f"Today's fake weather for {location} is _°")

func_dic = {get_current_weather.__name__: get_current_weather}
call_with_chat_completion(response, func_dic)
```

And you will get output like this:

```plaintext
Today's fake weather for Boston is _°
```

For more examples and a deeper dive into usage, please refer to the detailed documentation. 📘

## 🙋 Support

If you have any questions, run into any issues, or just need a little help, don't hesitate to reach out. We're here to help! 🤝

---

Whether you're a veteran developer or just getting started, GPTBase is a powerful tool for any text generation task. Give it a try today! 🌟