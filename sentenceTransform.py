from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import numpy as np
import os
import matplotlib.pyplot as plt


# Pulls all jobs from job-titles.txt and returns them as an array of strings
def getJobTitles(fileName) -> list[str]:
    with open(fileName) as f:
        jobs = [line.strip() for line in f if line.strip()]
    return jobs


# Handly class for everything sentence transformer related
class sentenceTransformer:
    def __init__(self, input="job-titles.txt", modelName="all-MiniLM-L6-v2") -> None:
        self.input = input
        self.modelName = modelName

        print(f"Loading model {modelName}...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        print(f"Loading data set from {input}...")
        self.dataSet = getJobTitles(input)

        # Check if we can load embeddings, or if we need to generate them
        if os.path.isfile(f"embeddings-({self.input}).npy"):
            print("Loading embeddings...")
            self.loadEmbeddings()
        else:
            print("Encoding embeddings...")
            self.embeddings = self.model.encode(self.dataSet)
            self.saveEmbeddings()

    def loadEmbeddings(self):
        if os.path.isfile(f"embeddings-({self.input}).npy"):
            self.embeddings = np.array(np.load(f"embeddings-({self.input}).npy"))
        else:
            print("No embeddings.npy exists!")

    def saveEmbeddings(self):
        np.save(f"embeddings-({self.input}).npy", self.embeddings)

    # Check the top k matches for a query
    def grabTopK(self, query, k=3):
        queryVector = self.model.encode([query])
        scores = cosine_similarity(queryVector, self.embeddings)[0]

        topIndices = np.argsort(scores)[-k:][::-1]

        for i in topIndices:
            print(self.dataSet[i], scores[i])

    def showGraph(self):
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(self.embeddings)
        plt.scatter(reduced[:, 0], reduced[:, 1])

        for i, job in enumerate(self.dataSet):
            plt.text(reduced[i, 0], reduced[i, 1], job, fontsize=8)

        plt.show()


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
