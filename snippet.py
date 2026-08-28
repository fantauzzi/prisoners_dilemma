import asyncio
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv


async def main():
    load_dotenv()
    prompt = ("If my barber shaves people, and only those people, who don't shave themsevelves, does he shave himself?")

    # llm = ChatOpenAI(model="gpt-5-nano", temperature=1, max_tokens=500)  # <== this gets an empty string in response.content
    # llm = ChatOpenAI(model="gpt-4.1-nano", temperature=1, max_tokens=500)  # <== This would work OK
    llm = ChatOpenAI(model="o4-mini", temperature=1, max_tokens=50)  # <== This would work OK
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    print(response.content)


asyncio.run(main())
