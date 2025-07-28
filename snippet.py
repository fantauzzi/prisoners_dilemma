#!/usr/bin/env python3
import os
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv

def main():
    load_dotenv()

    # Read your API key from the environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")

    # Initialize the ChatOpenAI client
    llm = ChatOpenAI(
        openai_api_key=api_key,
        model_name="gpt-4.1-nano",
        temperature=0.7,  # adjust for creativity (0.0–1.0)
        max_tokens=512     # adjust for response length
    )

    # Prompt the user
    prompt = input("Enter your prompt: ")

    # Send the prompt via 'invoke', passing the messages list as the positional argument
    response = llm.invoke([HumanMessage(content=prompt)])

    # Display the model's reply
    print("\nGPT-4.1 nano says:")
    print(response.content)


if __name__ == "__main__":
    main()
