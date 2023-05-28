# 🚀 GPTBase

Welcome to GPTBase! This Python library provides a straightforward and user-friendly interface to leverage OpenAI's powerful GPT-3.5-Turbo and GPT-4 models for text generation. 📚🔍

## ✨ Features

- 🎛️ Configurable parameters for chat completion.
- 💬 Support for both individual prompts and chat conversations.
- ⚡ Stream output option for faster response times.
- 📝 Built-in logging and rich printing for enhanced visualization.
- 🔄 Customizable model choice, including GPT-3.5-Turbo, GPT-4, and other variants.
- 📈 Easy model parameter adjustment.

## 📦 Installation

You can easily install the GPTBase library from PyPI with a single command line:

```bash
pip install gptbase
```

Once installed, import it in your Python script:

```python
from gptbase.basev2 import ChatAssistant, CompletionParameters
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

For more examples and a deeper dive into usage, please refer to the detailed documentation. 📘

## 🙋 Support

If you have any questions, run into any issues, or just need a little help, don't hesitate to reach out. We're here to help! 🤝

---

Whether you're a veteran developer or just getting started, GPTBase is a powerful tool for any text generation task. Give it a try today! 🌟