# GPTBase

Welcome to GPTBase, a Python library that provides a simple interface for using OpenAI's GPT-3.5-Turbo and GPT-4 models for text generation.

## Features

- Configurable parameters for chat completion.
- Support for both individual prompts and chat conversations.
- Stream output option for faster response times.
- Inbuilt logging and rich printing for better visualization.
- Customizable model choice, including GPT-3.5-Turbo, GPT-4, and other variants.
- Easy model parameter adjustment.

## Installation

You can install the GPTBase library from PyPI:

```
pip install gptbase
```

Then import it in your Python script:

```python
from gptbase.basev2 import ChatAssistant, CompletionParameters
```

## Usage

The library includes two main classes: `Assistant` for individual prompts and `ChatAssistant` for multi-turn chat conversations.

Here is a simple example of how to use the `ChatAssistant` class:

```python
from gptbase.basev2 import ChatAssistant, CompletionParameters

cm = CompletionParameters(stream=True)
assistant = ChatAssistant(memory_turns=3)
message = assistant.chat("I want to learn Python", cm)
```

For more examples and usage, please refer to the documentation.

## Support

If you have any questions or issues, feel free to contact us.

### Short Description

GPTBase is a versatile Python library for the OpenAI GPT-3.5-Turbo and GPT-4 models. It supports both individual prompts and chat conversations, with customizable parameters for output control. Other features include stream output for faster response times, built-in logging, and rich printing for better visualization.
