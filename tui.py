from sentenceTransform import sentenceTransformer
import numpy as np

jobTransformer = sentenceTransformer()

# An easy CLI for playing around
while True:
    cmd = input("""\n\n#############################
topk <k> <query> | Grab top k similar to input
graph            | Show a nice graph!
sanity           | Displays sanity checks
> """)

    cmdList = cmd.split(" ")
    print("--------------------")

    if cmdList[0] == "topk":
        jobTransformer.grabTopK(k=int(cmdList[1]), query=" ".join(cmdList[2:]))
    elif cmdList[0] == "check":
        jobTransformer.checkCosineSimilarity(query=" ".join(cmdList[2:]))
    elif cmdList[0] == "graph":
        jobTransformer.showGraph()
    elif cmdList[0] == "sanity":
        print(f"embeddings: {jobTransformer.embeddings.shape}")
        print("NaN / inf Checks")
        print(np.isnan(jobTransformer.embeddings).any())
        print(np.isinf(jobTransformer.embeddings).any())

    print("--------------------")
