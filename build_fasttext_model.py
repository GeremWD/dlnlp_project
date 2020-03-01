import fasttext
import os

if __name__ == '__main__':
    words = []
    with open("data/eng.train.txt", "r") as f:
        t = f.read().split()
        i = 0
        while i<len(t):
            words.append(t[i])
            i += 4

    with open("data/train_words.txt", "w") as train_words:
        train_words.write(" ".join(words))

    model = fasttext.train_unsupervised('data/train_words.txt', dim=32)
    model.save_model("data/fasttext_model.bin")
    os.remove("data/train_words.txt")
