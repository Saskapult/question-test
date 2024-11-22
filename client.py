#!/usr/bin/env python3

import sys
import os
from openai import OpenAI
import pydantic
from websockets.sync.client import connect

server_address = "ws://localhost:3528"
server_ws = None


def main():
	print("Hello world!")
	prompt_loop()


def beet_fact(query):
	global server_ws
	if server_ws is None:
		# print("Initialize connection to server...")
		try:
			server_ws = connect(server_address)
		except Exception as e:
			return f"{e}"		
	server_ws.send(query)
	response = server_ws.recv()
	return response


def prompt_loop():
	cowsay("Hello! Do you like beets? I Love them! Please ask me questions about beets.")
	while True:
		query = input(">")
		if query == "":
			break
		cowsay(beet_fact(query))
	if not server_ws is None:
		server_ws.close()


def cowsay(text):
	text = text.replace("'", "")
	text = text.replace('"', "")
	if os.system(f'cowsay "{text}"') != 0:
		print(text)


if __name__ == "__main__":
	main()

