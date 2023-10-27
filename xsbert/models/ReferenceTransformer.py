from sentence_transformers import models
import torch
import os
import json


class ReferenceTransformer(models.Transformer):
    '''adds reference to batch but does not subtract its embeddings after forward'''

    def forward(self, features):

        input_ids = features['input_ids']
        attention_mask = features['attention_mask']
        s = input_ids.shape[1]
        device = input_ids.device
        # TODO: generalize to arbitrary tokenizer
        ref_ids = torch.IntTensor([[0] + [1] * (s - 2) + [2]]).to(device)
        features['input_ids'] = torch.cat([input_ids, ref_ids], dim=0)
        if input_ids.shape[0] > 1:
            ref_att = torch.ones((1, s)).int().to(device)
            features['attention_mask'] = torch.cat([attention_mask, ref_att], dim=0)
        
        return super().forward(features)

        # emb = features['token_embeddings']

        # att = features['attention_mask']
        # if att.shape[0] > 1:
        #     att = att[:-1]

        # features.update({
        #     'token_embeddings': emb,
        #     'attention_mask': att
        # })        

        # return features
    
    
    @staticmethod
    def load(input_path: str):
        #Old classes used other config names than 'sentence_bert_config.json'
        for config_name in ['sentence_bert_config.json', 'sentence_roberta_config.json', 'sentence_distilbert_config.json', 'sentence_camembert_config.json', 'sentence_albert_config.json', 'sentence_xlm-roberta_config.json', 'sentence_xlnet_config.json']:
            sbert_config_path = os.path.join(input_path, config_name)
            if os.path.exists(sbert_config_path):
                break

        with open(sbert_config_path) as fIn:
            config = json.load(fIn)
        return ReferenceTransformer(model_name_or_path=input_path, **config)
    