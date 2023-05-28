from gptbase.basev2 import ChatAssistant, CompletionParameters


if __name__ == "__main__":
    cm = CompletionParameters(stream=True)
    memory_turns = int(input("memory turns: "))
    system_prompt = input("system prompt: ")
    chat = ChatAssistant(memory_turns, system_prompt)
    while True:
        message = input(">>> ")
        if message == "exit":
            break
        chat.chat(message, cm)
