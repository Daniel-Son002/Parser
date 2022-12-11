# code created following http://www.realworldnlpbook.com/blog/improving-sentiment-analyzer-using-elmo.html

import numpy as np
import torch
import torch.optim as optim
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.training.trainer import Trainer
from allennlp_models.classification.dataset_readers.stanford_sentiment_tree_bank import \
    StanfordSentimentTreeBankDatasetReader

from main import LstmClassifier
from predictor import SentenceClassifierPredictor

EMBEDDING_DIM = 128
HIDDEN_DIM = 128

from itertools import chain
    
def main():
    elmo_token_indexer = ELMoTokenCharactersIndexer()
    reader = StanfordSentimentTreeBankDatasetReader(
        token_indexers={'tokens': elmo_token_indexer})

    train_dataset = list(reader.read("\stanfordSentimentTreebank\trees\train.txt"))
    dev_dataset = list(reader.read("\stanfordSentimentTreebank\trees\dev.txt"))

    options_file = ('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo'
                    '/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json')
    weight_file = ('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo'
                   '/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5')

    elmo_embedder = ElmoTokenEmbedder(options_file, weight_file)

    vocab = Vocabulary.from_instances(train_dataset + dev_dataset,
                                      min_count={'tokens': 3})
    # vocab = Vocabulary.from_instances(chain(train_dataset , dev_dataset),
    #                                   min_count={'tokens': 3})
    
    embedder = BasicTextFieldEmbedder({"tokens": elmo_embedder})

    elmo_embedding_dim = 256
    lstm = PytorchSeq2VecWrapper(
        torch.nn.LSTM(elmo_embedding_dim, HIDDEN_DIM, batch_first=True))

    model = LstmClassifier(embedder, lstm, vocab)
    optimizer = optim.Adam(model.parameters())

    iterator = BucketIterator(batch_size=32, sorting_keys=[("tokens", "num_tokens")])

    iterator.index_with(vocab)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=dev_dataset,
                      patience=10,
                      num_epochs=20)

    trainer.train()

    tokens = ['This', 'is', 'the', 'best', 'movie', 'ever', '!']
    predictor = SentenceClassifierPredictor(model, dataset_reader=reader)
    logits = predictor.predict(tokens)['logits']
    label_id = np.argmax(logits)

    print(model.vocab.get_token_from_index(label_id, 'labels'))

if __name__ == '__main__':
    main()
