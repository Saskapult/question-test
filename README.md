# question-test
A project wherein I play with a RAG-based question and answer bot. 

```
 ________________________________________
/ Hello! Do you like beets? I Love them! \
\ Please ask me questions about beets.   /
 ----------------------------------------
        \   ^__^
         \  (oo)\_______
            (__)\       )\/\
                ||----w |
                ||     ||
>What kind of beet is the most tasty?
```

## Setup
This project requires the `cowsay` command-line utility. 

Python package requirements are found in `requirements.txt`. 

An OpenAI API key must be supplied using the `OPENAI_API_KEY` environment variable or as the first argument to the program. 

The system draws data from a collection of `.pdf` files found in `./data`. 
Download some slides and put them in there. 

Run `python3 server.py` and then (in another terminal) `python3 client.py`. 
A cow will greet you and ask for questions about beets, but they actually want questions about kelp. 
