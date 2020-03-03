# Adapting Transformer Encoder for Named Entity Recognition

Here is our attempt to reproduce part of the results from the paper https://arxiv.org/pdf/1911.04474.pdf


## Usage

To install fasttext :
```
git clone https://github.com/facebookresearch/fastText.git  
cd fastText 
make
```

To produce the fasttext model :
`python build_fasttext_model.py`

To preprocess the data and build formatted datasets in the data folder :
`python preprocessing.py`

To train the model :
`python train.py`
