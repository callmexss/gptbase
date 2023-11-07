import click
from gptbase.base import Chat, CompletionParameters

@click.command()
@click.option('--memory-turns', default=3, help='Number of memory turns.')
@click.option('--system-prompt', default='', help='System prompt.')
def chat(memory_turns, system_prompt):
    cm = CompletionParameters(stream=True)
    chat = Chat(memory_turns, system_prompt)
    print("Welcome to Chat build with GPTBase!")
    print("You can exit by hit ctrl c or typing `exit`.")
    while True:
        try:
            message = input(">>> ")
            if message == "exit":
                break
            chat.chat(message, cm)
        except KeyboardInterrupt:
            break
        except Exception as err:
            print(err)


if __name__ == "__main__":
    chat()
