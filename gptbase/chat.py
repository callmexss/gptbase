import click
from gptbase.basev2 import ChatAssistant, CompletionParameters

@click.command()
@click.option('--memory-turns', default=3, help='Number of memory turns.')
@click.option('--system-prompt', default='', help='System prompt.')
def chat(memory_turns, system_prompt):
    cm = CompletionParameters(stream=True)
    chat = ChatAssistant(memory_turns, system_prompt)
    while True:
        message = input(">>> ")
        if message == "exit":
            break
        chat.chat(message, cm)

if __name__ == "__main__":
    chat()
