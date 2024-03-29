from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="gptbase",
    version="0.3.0",
    author="callmexss",
    author_email="callmexss@126.com",
    description="A package for simplified interaction with OpenAI's GPT-3 and GPT-4 models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/callmexss/gptbase",
    packages=find_packages(include=["gptbase"]),
    install_requires=[
        "openai==1.1.1",
        "rich==13.0.1",
        "tiktoken==0.5.1",
        "click==8.0.3",
        "iso-639==0.4.5",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "gptbase-chat=gptbase.chat:chat",
        ],
    },
)
