from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
import os
import gzip
import csv

from xsbert.models import ShiftingReferenceTransformer, XSTransformer, DotSimilarityLoss
from xsbert.utils import load_sts_data


# data
sts_dataset_path = '../../data/similarity/stsbenchmark.tsv.gz'

if not os.path.exists(sts_dataset_path):
    util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)

# training config
# model_name = 'sentence-transformers/all-mpnet-base-v2'
model_name ='sentence-transformers/all-distilroberta-v1'
model_save_path = '../xs_models/xs_distilroberta'
train_batch_size = 16 
num_epochs = 5

if not os.exists(model_save_path):
    os.makedirs(model_save_path)

# model
word_embedding_model = ShiftingReferenceTransformer(model_name)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = XSTransformer(
    modules=[word_embedding_model, pooling_model],
    device='cuda:0'
    )

# dataloading
train_samples, dev_samples, test_samples = load_sts_data(sts_dataset_path)
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')

# training
train_loss = DotSimilarityLoss(model=model)
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=math.ceil(len(train_dataloader) * num_epochs  * 0.1),
          output_path=model_save_path)

# loading model checkpoint and running evaluation
model = XSTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
test_evaluator(model, output_path=model_save_path)