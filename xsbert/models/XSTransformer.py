import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing import Tuple

from ..utils import input_to_device
from . import hooks


class XSTransformer(SentenceTransformer):

    def forward(self, features: dict):
        features = super().forward(features)
        emb = features['sentence_embedding']
        att = features['attention_mask']
        features.update({
            'sentence_embedding': emb[:-1],
            'attention_mask': att[:-1]
        })
        features['reference'] = emb[-1]
        features['reference_att'] = att[-1]
        return features

    def init_attribution_to_layer(self, idx: int, N_steps: int):
        raise NotImplementedError()

    def reset_attribution(self):
        raise NotImplementedError()

    def tokenize_text(self, text: str):
        tokens = self[0].tokenizer.tokenize(text)
        tokens = [t[1:] if t[0] in ['Ġ', 'Â'] else t for t in tokens]
        tokens = ['CLS'] + tokens + ['EOS']
        return tokens

    def _compute_integrated_jacobian(
            self, 
            embedding: torch.Tensor, 
            intermediate: torch.Tensor,
            move_to_cpu: bool = True,
            verbose: bool = True
            ):
        D = self[0].get_word_embedding_dimension()
        jacobi = []
        retain_graph = True
        for d in tqdm(range(D), disable = not verbose):
            if d == D - 1: 
                retain_graph = False
            grads = torch.autograd.grad(
                list(embedding[:, d]), intermediate, retain_graph=retain_graph
                )[0].detach()
            if move_to_cpu:
                grads = grads.cpu()
            jacobi.append(grads)
        J = torch.stack(jacobi) / self.N_steps
        J = J[:, :-1, :, :].sum(dim=1)
        return J
        
    def explain_similarity(
            self, 
            text_a: str, 
            text_b: str, 
            sim_measure: str = 'cos',
            return_lhs_terms: bool = False,
            move_to_cpu: bool = False,
            verbose: bool = True,
            compress_embedding_dim: bool = True
            ):

        assert sim_measure in ['cos', 'dot'], f'invalid argument for sim_measure: {sim_measure}'

        self.intermediates.clear()
        device = self[0].auto_model.embeddings.word_embeddings.weight.device

        #TODO: this should be a method
        inpt_a = self[0].tokenize([text_a])
        input_to_device(inpt_a, device)
        features_a = self.forward(inpt_a)
        emb_a = features_a['sentence_embedding']
        interm_a = self.intermediates[0]
        J_a = self._compute_integrated_jacobian(emb_a, interm_a, move_to_cpu=move_to_cpu, verbose=verbose)
        D, Sa, Da = J_a.shape
        J_a = J_a.reshape((D, Sa * Da))

        da = interm_a[0] - interm_a[-1]
        da = da.reshape(Sa * Da, 1).detach()

        inpt_b = self[0].tokenize([text_b])
        input_to_device(inpt_b, device)
        features_b = self.forward(inpt_b)
        emb_b = features_b['sentence_embedding']
        interm_b = self.intermediates[1]
        J_b = self._compute_integrated_jacobian(emb_b, interm_b, move_to_cpu=move_to_cpu, verbose=verbose)
        _, Sb, Db = J_b.shape
        J_b = J_b.reshape((D, Sb * Db))

        db = interm_b[0] - interm_b[-1]
        db = db.reshape(Sb * Db, 1).detach()

        J = torch.mm(J_a.T, J_b)
        da = da.repeat(1, Sb * Db)
        db = db.repeat(1, Sa * Da)
        if move_to_cpu:
            da = da.detach().cpu()
            db = db.detach().cpu()
            emb_a = emb_a.detach().cpu()
            emb_b = emb_b.detach().cpu()
        A = da * J * db.T
        if sim_measure == 'cos':
            A = A / torch.norm(emb_a[0]) / torch.norm(emb_b[0])
        A = A.reshape(Sa, Da, Sb, Db)
        if compress_embedding_dim:
            A = A.sum(dim=(1, 3))
        A = A.detach().cpu()

        tokens_a = self.tokenize_text(text_a)
        tokens_b = self.tokenize_text(text_b)

        if return_lhs_terms:
            ref_a = features_a['reference']
            ref_b = features_b['reference']
            if move_to_cpu:
                ref_a = ref_a.detach().cpu()
                ref_b = ref_b.detach().cpu()
            if sim_measure == 'cos':
                score = torch.cosine_similarity(emb_a[0].unsqueeze(0), emb_b[0].unsqueeze(0)).item()
                ref_emb_a = torch.cosine_similarity(emb_a[0].unsqueeze(0), ref_b.unsqueeze(0)).item()
                ref_emb_b = torch.cosine_similarity(emb_b[0].unsqueeze(0), ref_a.unsqueeze(0)).item()
                ref_ref = torch.cosine_similarity(ref_a.unsqueeze(0), ref_b.unsqueeze(0)).item()
            elif sim_measure == 'dot':
                score = torch.dot(emb_a[0], emb_b[0]).item()
                ref_emb_a = torch.dot(emb_a[0], ref_b).item()
                ref_emb_b = torch.dot(emb_b[0], ref_a).item()
                ref_ref = torch.dot(ref_a, ref_b).item()
            return A, tokens_a, tokens_b, score, ref_emb_a, ref_emb_b, ref_ref
        else:
            return A, tokens_a, tokens_b
        

    def explain_by_decomposition(self, text_a: str, text_b: str, normalize: bool = False):
        
        device = self[0].auto_model.embeddings.word_embeddings.weight.device
        
        inpt_a = self[0].tokenize([text_a])
        input_to_device(inpt_a, device)
        emb_a = self.forward(inpt_a)['token_embeddings'][0]
        if normalize:
            emb_a = emb_a / torch.sum(emb_a)

        inpt_b = self[0].tokenize([text_b])
        input_to_device(inpt_b, device)
        emb_b = self.forward(inpt_b)['token_embeddings'][0]
        if normalize:
            emb_b = emb_b / torch.sum(emb_b)

        A = torch.mm(emb_a, emb_b.t()).detach().cpu()
        A = A / emb_a.shape[0] / emb_b.shape[0]

        tokens_a = self.tokenize_text(text_a)
        tokens_b = self.tokenize_text(text_b)

        return A, tokens_a, tokens_b


    def score(self, texts: Tuple[str]):
        self.eval()
        with torch.no_grad():
            inputs = [self[0].tokenize([t]) for t in texts]
            for inpt in inputs:
                input_to_device(inpt, self.device)
            embeddings = [self.forward(inpt)['sentence_embedding'] for inpt in inputs]
            s = torch.dot(embeddings[0][0], embeddings[1][0]).cpu().item()
            del embeddings
            torch.cuda.empty_cache()
        return s


class XSRoberta(XSTransformer):

    def init_attribution_to_layer(self, idx: int, N_steps: int):
        
        if hasattr(self, 'hook') and self.hook is not None:
            raise AttributeError('a hook is already registered')
        try:
            assert idx < len(self[0].auto_model.encoder.layer), f'the model does not have a layer {idx}'
        except AttributeError:
            raise AttributeError('The encoder model is not supported')
        
        self.N_steps = N_steps
        self.intermediates = []
        self.hook = self[0].auto_model.encoder.layer[idx].register_forward_pre_hook(
            hooks.roberta_interpolation_hook(N=N_steps, outputs=self.intermediates)
        )

    def reset_attribution(self):
        if hasattr(self, 'hook'):
            self.hook.remove()
            self.hook = None
        else:
            print('No hook has been registered.')


class XSMPNet(XSTransformer):

    def init_attribution_to_layer(self, idx: int, N_steps: int):
        
        if hasattr(self, 'hook') and self.interpolation_hook is not None:
            raise AttributeError('a hook is already registered')
        try:
            assert idx < len(self[0].auto_model.encoder.layer), f'the model does not have a layer {idx}'
        except AttributeError:
            raise AttributeError('The encoder model is not supported')
        
        self.N_steps = N_steps
        self.intermediates = []
        self.interpolation_hook = self[0].auto_model.encoder.layer[idx].register_forward_pre_hook(
            hooks.mpnet_interpolation_hook(N=N_steps, outputs=self.intermediates)
        )
        self.reshaping_hooks = []
        for l in range(idx + 1, len(self[0].auto_model.encoder.layer)):
            handle = self[0].auto_model.encoder.layer[l].register_forward_pre_hook(
                hooks.mpnet_reshaping_hook(N=N_steps)
            )
            self.reshaping_hooks.append(handle)

    def reset_attribution(self):
        if hasattr(self, 'interpolation_hook'):
            self.interpolation_hook.remove()
            del self.interpolation_hook
            for hook in self.reshaping_hooks:
                hook.remove()
            del self.reshaping_hooks


if __name__ == '__main__':

    from sentence_transformers.models import Pooling

    from utils import input_to_device
    from .ReferenceTransformer import ReferenceTransformer
    from .XSTransformer import XSMPNet

    # model = ReferenceTransformer('sentence-transformers/all-mpnet-base-v2')
    # pooling = Pooling(model.get_word_embedding_dimension())
    # explainer = XSMPNet(modules=[model, pooling], device='cuda:2')

    model_path = '../../experiments/x_smpnet_cos_02/'
    explainer = XSMPNet(model_path)
    explainer.to(torch.device('cuda:1'))

    texta = 'This concept generalizes poorly.'
    textb = 'The shown principle generalizes to other areas.'    

    explainer.init_attribution_to_layer(idx=11, N_steps=250)
    A, ta, tb, ab, ra, rb, rr = explainer.explain_similarity(texta, textb, move_to_cpu=True, return_lhs_terms=True, sim_measure='cos')
    print(A.sum().item(), ab - ra - rb + rr)
    print(ab, ra, rb, rr)