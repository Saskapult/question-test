#!/usr/bin/env python3

import sys
import os
from openai import OpenAI
import asyncio
from websockets.asyncio.server import serve
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import pickle 
from tika import parser
import glob
from dataclasses import dataclass

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


# A chunk of data for the RAG system to use
class DataChunk:
	source: str    # slides.pdf/site.html
	index: str     # slide/paragraph
	embedding: str # embedding, retrieved from cache
	content: str   # text of the section

	def __init__(self, source, index, content):
		self.source = source
		self.index = index
		self.content = content

		h = hashlib.md5(self.content.encode("utf-8")).hexdigest()
		if os.path.isfile(f"./embeddings/{h}"):
			print("Load embedding from file...")
			with open(f"./embeddings/{h}", "rb") as f:
				self.embedding = pickle.load(f)
		else:
			print(f"Create embedding...")
			embedding = client.embeddings.create(
				input=self.content,
				model="text-embedding-3-small",
			).data[0].embedding

			if not os.path.exists("./embeddings"):
				os.makedirs("./embeddings")
			with open(f"./embeddings/{h}", "wb") as f:
				pickle.dump(embedding, f)

	# Indexes an entire pdf file
	# Todo: windowed and per-slide indexing
	def index_pdf(path):
		print(f"Reading '{path}'")
		raw = parser.from_file(path)
		return [DataChunk(path, ":full", raw["content"])]
	

class BeetRAG:
	def __init__(self, chunks):
		self.chunks = chunks
	
	# Returns most similar chunks in descending order of similarity
	def similar_chunks(self, query, cutoff):
		embedding = client.embeddings.create(
			input=query,
			model="text-embedding-3-small",
		).data[0].embedding
		embeddings = [c.embedding for c in self.chunks]
		similarities = cosine_similarity(np.array(embeddings), np.array(embedding).reshape(1,-1)).flatten()
		ss = similarities.argsort()
		print(similarities)
		print(ss)
		print(similarities.argmax())
		cut = ss[similarities >= cutoff]
		print(cut)
		if len(cut) > 0:
			return self.chunks[cut]
		else:
			return []

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

	# Tests that a response contains only information found in the source
	def hallucination_test(self, response, source):
		hallucination_response = client.beta.chat.completions.parse(
			model="gpt-4o-mini",
			messages=[
				{"role": "system", "content": "Decide if the response contains only information found in the source text. Output 'yes' or 'no'. Output 'yes' if the response contains only information found in the source text. Output 'no' if response contains information not found in the source text."},
				{"role": "user", "content": f"Response:\n{response}\nSource text:\n{source}"},
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
	
	# Attempts to answer a query from the user
	def query(self, query):
		hallucination_retry_attempts = 4
		for retry_i in range(1, hallucination_retry_attempts+1):

			similar_documents = self.similar_chunks(query, 0.4)
			if len(similar_documents) == 0:
				return "Error: No relevant documents found!"
			# Assume that the document is relevant if it has sufficient cosine similarity 
			print(f"Found {len(similar_documents)} similar documents")
			sources = ", ".join([f"{c.source}:{c.index}" for c in similar_documents])
			source = "\n".join([c.content for c in similar_documents])

			# Use the conent to make an answer? 
			response = client.beta.chat.completions.parse(
				model="gpt-4o-mini",
				messages=[
					{"role": "system", "content": "Answer questions about kelp. Do not answer questions that are not about kelp."},
					{"role": "assistant", "content": source},
					{"role": "user", "content": f"{query}"},
				],
			)
			answer = response.choices[0].message.content
			print(f"Query: {query}")
			print(f"Response: {answer}")

			if self.hallucination_test(answer, source):
				print(f"Hallucination detected, retrying ({retry_i})")
				query = self.rewrite_query(query)
				continue
			else:
				return response2 + f"\nSources: {sources}"
		return "Error: Response re-generation maximum reached!"


def main():
	print(f"Loading database...")
	chunks = []
	for pdf in glob.glob("data/*.pdf"):
		for c in DataChunk.index_pdf(pdf):
			chunks.append(c)

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

