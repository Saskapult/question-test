#!/usr/bin/env python3

import sys
import os
from openai import OpenAI
import pydantic
import asyncio
from websockets.asyncio.server import serve
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import pickle 

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

client = OpenAI(api_key=api_key)
rag = None

class QueryResponse(pydantic.BaseModel):
	text: str

class BeetRAG:
	def __init__(self, database):
		print(f"Loading database...")
		# Simple paragraph splitting
		f = open(database, "r")
		paragraphs = f.read().split("\n\n")
		f.close()

		# Embeddings are cached to reduce API calls 
		self.chunks = paragraphs
		h = hashlib.md5(database.encode("utf-8")).hexdigest()
		print(f"Hash is {h}")
		if os.path.isfile(f"./embeddings/{h}"):
			print("Load embeddings from file...")
			with open(f"./embeddings/{h}", "rb") as f:
				self.embeddings = pickle.load(f)
		else:
			print(f"Create {len(self.chunks)} embeddings...")
			self.embeddings = [client.embeddings.create(
				input=c,
				model="text-embedding-3-small",
			).data[0].embedding for c in self.chunks]

			if not os.path.exists("./embeddings"):
				os.makedirs("./embeddings")
			with open(f"./embeddings/{h}", "wb") as f:
				pickle.dump(self.embeddings, f)

		print("Done init!")
	
	def chunk_of(self, query):
		embedding = client.embeddings.create(
			input=query,
			model="text-embedding-3-small",
		).data[0].embedding
		i = cosine_similarity(np.array(self.embeddings), np.array(embedding).reshape(1,-1)).argmax()
		return self.chunks[i]
	
	def query(self, query):
		chunk = self.chunk_of(query)
		response = client.beta.chat.completions.parse(
			model="gpt-4o-mini",
			messages=[
				{"role": "system", "content": "Answer questions about beets. Do not answer questions that are not about beets."},
				{"role": "assistant", "content": chunk},
				{"role": "user", "content": f"{query}"},
			],
		)
		response = response.choices[0].message.content
		print(f"Query: {query}")
		print(f"Chunk: {chunk}")
		print(f"Response: {response}")
		return response


def main():
	print("Hello world!")

	global rag
	rag = BeetRAG("beet_facts")

	asyncio.run(query_serve_loop())


async def query_serve_loop():
	async with serve(query_serve, "localhost", 3528) as server:
		await asyncio.Future()


async def query_serve(websocket):
	global rag
	async for message in websocket:
		bf = rag.query(message)
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

