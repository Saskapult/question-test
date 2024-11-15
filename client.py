#!/usr/bin/env python3

import sys
import os
from openai import OpenAI
import pydantic

api_key = ""
if len(sys.argv) > 1:
	print("Using argument API key")
	api_key = sys.argv[1]
elif os.environ.contains("OPENAI_API_KEY"):
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
	prompt_loop()


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


def prompt_loop():
	cowsay("Hello! Do you like beets? I Love them! Please ask me questions about beets.")
	while True:
		query = input(">")
		if query == "":
			break
		cowsay(beet_fact(query))


def cowsay(text):
	os.system(f'cowsay "{text}"')


if __name__ == "__main__":
	main()

