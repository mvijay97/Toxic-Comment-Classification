# Toxic-Comment-Classification
We use a <b> Hierarchical Attention Network (HAN) </b> to identify toxic comments using the <a href='https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data'> Wikipedia Toxic Comment Dataset</a>. The model identifyies six classes of toxicity - 
* toxic
* severe_toxic
* obscene
* threat
* insult
* identity_hate

## How does it work?
The HAN implemented for this problem operates on a two level hierarchy, one that operates on the word level, and the other that operates on the sentence level. Each level employs a bidirectional Recurrent Neural Network (RNN) and an attention mechanism to operate on its inputs. <b> Word Level </b> representations are fed into the <b> Sentence Level </b> hierarchy which yields Sentence level representations. Here is a description of the flow - 

1. Sentence Level input is fed into the word level hierarchy which operates on each word in each sentence of the review. 
2. Word level embeddings are generated using pre-trained Glove embeddings. 
3. The embeddings are fed into a word level bidirectional RNN composed of GRU cells (no. of neurons = 150).
4. The concatenated outputs from the BiRNN are fed into a word level attention layer (attention size = 50).
5. The attention outputs of each word are stacked to form a matrix with all the word level hierarchy outputs for a sentence.
6. The matrix is fed into the sentence level BiRNN (number of neurons = 225).
7. The concatenated outputs are fed into a sentence level attention layer (attention size = 50).
8. The outputs of the sentence level attention layer are fed into a fully connected dense layer (no. of neurons = 6). 
9. Binary Cross Entropy loss computed.

## How to Run

* Download the dataset at <a href='https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data'> Wikipedia Toxic Comment Dataset</a>
* Download the pretrained Glove embeddings at <a href='https://nlp.stanford.edu/projects/glove/'> Glove.6B </a>
* Run the HAN-WordLevel.ipynb notebook to train and test a network with only word level attention 
* Run the HAN-SentenceLevel.ipynb notebook to train and test a network with word and sentence level attention
