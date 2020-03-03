
import fasttext
import numpy as np

def preprocess(model_name, input_name, embeddings_name, masks_name, targets_name, max_sentence_len):
    model = fasttext.load_model(model_name)
    d = model.get_dimension()

    class_to_int = {'O':0, 'B-ORG':1, 'I-ORG':1, 'B-LOC':2, 'I-LOC':2, 'B-MISC':3, 'I-MISC':3, 'B-PER':4, 'I-PER':4}

    embeddings = []
    targets = []

    sentence_embeddings = []
    sentence_targets = []

    with open(input_name, "r") as f:
        for line in f:
            if line == '\n':
                if len(sentence_embeddings) <= max_sentence_len and len(sentence_embeddings) > 0:
                    embeddings.append(sentence_embeddings)
                    targets.append(sentence_targets)
                sentence_embeddings = []
                sentence_targets = []
            else:
                words = line.split()
                assert(len(words) == 4)
                sentence_embeddings.append(model.get_word_vector(words[0]))
                sentence_targets.append(class_to_int[words[3]])

    max_len = max(len(s) for s in targets)

    embeddings_tensor = np.zeros((len(targets), max_len, d))
    masks_tensor = np.full((len(targets), max_len), True, dtype='bool')
    targets_tensor = np.full((len(targets), max_len), -1, dtype='int')
    for i in range(len(targets)):
        assert len(embeddings[i]) > 0, i
        embeddings_tensor[i, :len(embeddings[i]), :] = embeddings[i]
        masks_tensor[i, len(embeddings[i]):] = False
        targets_tensor[i, :len(embeddings[i])] = targets[i]

    np.save(embeddings_name, embeddings_tensor)
    np.save(masks_name, masks_tensor)
    np.save(targets_name, targets_tensor)

if __name__ == '__main__':
    preprocess("data/fasttext_model.bin", "data/conll/eng.train.txt", "data/train_embeddings", "data/train_masks", "data/train_targets", 32)
    preprocess("data/fasttext_model.bin", "data/conll/eng.testa.txt", "data/testa_embeddings", "data/testa_masks", "data/testa_targets", 32)
    preprocess("data/fasttext_model.bin", "data/conll/eng.testb.txt", "data/testb_embeddings", "data/testb_masks", "data/testb_targets", 32)

