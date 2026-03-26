# CSIT314 Sentence Transformer Backend
A simple way to quickly check what job titles are similar to one another via vectorisation!

## How to Use
Firstly, make sure dependencies are install, either on your system Python or a .venv: `pip install -r req.txt`
Running the `tui.py` gives a really nice overview of what the class is capable of.
Almost any .txt can be loaded through the class as well! Not just the default *job-titles.txt*.

## The Sentence Transformer Class
### init
Normal class initialisation stuff.
- **input**: The .txt file to load into embeddings, will store them for later.
- **modeName**: The sentence transformer to use, defaults to *all-MiniLM-L6-v2*
### grabTopK
Pulls the top k related sentences to a query
- **k**: How many related sentences to pull.
- **query**: The sentence to prompt into the sentence transformer.
### showGraph
Shows a nice graph of all data points! Uses decomposition to reduce dimensionality to only 2, so most nuance is lost. Will be messy with lots of data though...
