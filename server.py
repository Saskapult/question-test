#!/usr/bin/env python3

import sys
import os
from openai import OpenAI
import pydantic
import asyncio
from websockets.asyncio.server import serve

api_key = ""
if len(sys.argv) > 1:
	print("Using argument API key")
	api_key = sys.argv[1]
elif os.environ.get("OPENAI_API_KEY") is not None:
	print("Using environment API key")
	api_key = os.environ.get("OPENAI_API_KEY")
else:
	print("No API key supplied!")
	exit(1)

# 4o
client = OpenAI(api_key=api_key)

class QueryResponse(pydantic.BaseModel):
	text: str


def main():
	print("Hello world!")
	asyncio.run(query_serve_loop())


async def query_serve_loop():
	async with serve(query_serve, "localhost", 3528) as server:
		await asyncio.Future()


async def query_serve(websocket):
	async for message in websocket:
		bf = beet_fact(message)
		print(f"{message} -> {bf}")
		await websocket.send(bf)


def beet_fact(query):
	f = open("beet_facts", "r")
	lines = f.readlines()
	f.close()

	lines = [line for line in lines if not line.startswith("#")]
	lines = [line.strip() for line in lines]
	lines = [line for line in lines if line != ""]
	data = "\n".join(lines)

	response = client.beta.chat.completions.parse(
		model="gpt-4o-mini",
		messages=[
			{"role": "system", "content": "Answer questions about beets."},
			{"role": "user", "content": f"""
Use the article below to answer the subsequent question. If the answer cannot be found, write "I don't know."

Article:
{data}

Question:
{query}
			""".strip()},
		],
		response_format=QueryResponse,
	)

	return response.choices[0].message.content


if __name__ == "__main__":
	main()

