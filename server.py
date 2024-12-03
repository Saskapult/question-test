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
# import tika 
from tika import parser
import glob

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
	def __init__(self, chunks):
		# Embeddings are cached to reduce API calls 
		self.chunks = chunks
		self.embeddings = []
		# print(f"Hash is {h}")
		for chunk in self.chunks:
			h = hashlib.md5(chunk.encode("utf-8")).hexdigest()
			embedding = None
			if os.path.isfile(f"./embeddings/{h}"):
				print("Load embedding from file...")
				with open(f"./embeddings/{h}", "rb") as f:
					embedding = pickle.load(f)
			else:
				print(f"Create embedding...")
				embedding = client.embeddings.create(
					input=chunk,
					model="text-embedding-3-small",
				).data[0].embedding

				if not os.path.exists("./embeddings"):
					os.makedirs("./embeddings")
				with open(f"./embeddings/{h}", "wb") as f:
					pickle.dump(embedding, f)
			self.embeddings.append(embedding)

		print("Done init!")
	
	def chunk_of(self, query):
		embedding = client.embeddings.create(
			input=query,
			model="text-embedding-3-small",
		).data[0].embedding
		c = cosine_similarity(np.array(self.embeddings), np.array(embedding).reshape(1,-1))
		i = c.argmax()
		return self.chunks[i]

	def rewrite_query(self, query):
		response = client.beta.chat.completions.parse(
			model="gpt-4o-mini",
			messages=[
				{"role": "system", "content": "Re-write questions."},
				{"role": "user", "content": query},
			],
		)
		# Check that it is the same question? 
		return response.choices[0].message.content

	def hallucination_test(self, response, chunk):
		hallucination_response = client.beta.chat.completions.parse(
			model="gpt-4o-mini",
			messages=[
				{"role": "system", "content": "Decide if the response contains only information found in the source text. Output 'yes' or 'no'. Output 'yes' if the response contains only information found in the source text. Output 'no' if response contains information not found in the source text."},
				{"role": "assistant", "content": chunk},
				{"role": "user", "content": f"Response:\n{response}\nSource text:\n{chunk}"},
			],
		)
		if hallucination_response.choices[0].message.content.lower() == "no":
			print("Checker detects hallucination!")
			return True
		if hallucination_response.choices[0].message.content.lower() == "yes":
			return False
		else:
			print(f"Unintended checker output: '{hallucination_response.choices[0].message.content}'")
			return False
	
	def query(self, query):

		for k in range(0, 4):
			# Find relevant document(s)
			embedding = client.embeddings.create(
				input=query,
				model="text-embedding-3-small",
			).data[0].embedding
			c = cosine_similarity(np.array(self.embeddings), np.array(embedding).reshape(1,-1))
			i = c.argmax()
			print(f"Chunk {i} similarity {c[i]}")
			if c[i] < 0.2:
				return "Error: Low document relevancy!"
			# Assume that the document is relevant if it has sufficient cosine similarity 

			chunk = self.chunks[i]
			response = client.beta.chat.completions.parse(
				model="gpt-4o-mini",
				messages=[
					{"role": "system", "content": "Answer questions about seaweed. Do not answer questions that are not about seaweed."},
					{"role": "assistant", "content": chunk},
					{"role": "user", "content": f"{query}"},
				],
			)
			response2 = response.choices[0].message.content
			print(f"Query: {query}")
			# print(f"Chunk: {chunk}")
			print(f"Response: {response2}")
			if len(response.choices) > 1:
				print(f"Responses ({len(response.choices)}):")
				for response in response.choices:
					print("\t", response.message.content)

			if self.hallucination_test(response2, chunk):
				print(f"Hallucination detected, retrying ({k})")
				query = self.rewrite_query(query)
				continue
			else:
				return response2
		return "Error: Response re-generation maximum reached!"


def main():
	print("Hello world!")

	print(f"Loading database...")
	# # Simple paragraph splitting
	# f = open(database, "r")
	# paragraphs = f.read().split("\n\n")
	# f.close()

	# Each pdf gets an embedding 
	# How can we direct the user to a slide in the deck?
	# Each WINDOW of n slides gets an embedding?

	chunks = []
	for pdf in glob.glob("seaweed_data/*.pdf"):
		print(f"Read {pdf}")
		raw = parser.from_file(pdf)
		chunks.append(raw["content"])
	# for chunk in chunks:
	# 	print("-----------------")
	# 	print(chunk)
	# exit(0)

	global rag
	rag = BeetRAG(chunks)

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


if __name__ == "__main__":
	main()

