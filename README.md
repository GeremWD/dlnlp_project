# Adapting Transformer Encoder for Named Entity Recognition

Here is our attempt to reproduce part of the results from the paper https://arxiv.org/pdf/1911.04474.pdf


## Usage

To preprocess the data and build formatted datasets in the data folder :  
`python preprocessing.py fasttext_model`  
where `fasttext_model` is a binary model produced with the fasttext module.  
Such a model can be downloaded on the fasttext website : https://fasttext.cc/docs/en/pretrained-vectors.html.

It can also be produced from scratch from the words inside the training set by installing fasttext and running the `build_fasttext_model.py` script :  
```
git clone https://github.com/facebookresearch/fastText.git  
cd fastText 
make
cd ..
python build_fasttext_model.py
```

To train the model (the preprocessing shoud have been done first) :  
`python train.py`  
The file is made to use cuda but you can easily change that by removing the several `.cuda()` in `train.py`.
