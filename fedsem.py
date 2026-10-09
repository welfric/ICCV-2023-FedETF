import os
import torch

##############################################################################
# FedSem: fixed semantic classifier built from CLIP text embeddings
# Algorithm 1 (server, once): semantic simplex construction
##############################################################################

# CLIP's prompt ensemble for CIFAR-10/100
CLIP_TEMPLATES = [
    'a photo of a {}.',
    'a blurry photo of a {}.',
    'a black and white photo of a {}.',
    'a low contrast photo of a {}.',
    'a high contrast photo of a {}.',
    'a bad photo of a {}.',
    'a good photo of a {}.',
    'a photo of a small {}.',
    'a photo of a big {}.',
    'a photo of the {}.',
    'a blurry photo of the {}.',
    'a black and white photo of the {}.',
    'a low contrast photo of the {}.',
    'a high contrast photo of the {}.',
    'a bad photo of the {}.',
    'a good photo of the {}.',
    'a photo of the small {}.',
    'a photo of the big {}.',
]


def get_class_names(train_set):
    '''
    class names c_k in label order
    '''
    if hasattr(train_set, 'label_texts'):
        # TinyImageNet: map the WordNet ids to their first lemma via words.txt
        with open(os.path.join(train_set.root, 'words.txt'), 'r') as fp:
            words = dict(line.rstrip('\n').split('\t', 1) for line in fp)
        return [words[wnid].split(',')[0].strip() for wnid in train_set.label_texts]
    return [name.replace('_', ' ') for name in train_set.classes]


@torch.no_grad()
def encode_class_names(class_names, templates, backend='openai', model_name=None, device='cpu'):
    '''
    prompt ensemble: u_k = 1/J sum_j E_txt(pi_j(c_k)) / ||E_txt(pi_j(c_k))||
    '''
    if backend == 'openai':
        import clip
        model, _ = clip.load(model_name or 'ViT-B/32', device=device)
        encode = lambda prompts: model.encode_text(clip.tokenize(prompts).to(device))
    elif backend == 'hf':
        from transformers import CLIPModel, CLIPTokenizer
        model_name = model_name or 'openai/clip-vit-base-patch32'
        model = CLIPModel.from_pretrained(model_name).to(device).eval()
        tokenizer = CLIPTokenizer.from_pretrained(model_name)
        encode = lambda prompts: model.text_projection(model.text_model(**tokenizer(prompts, padding=True, return_tensors='pt').to(device)).pooler_output)
    else:
        raise ValueError('Undefined clip backend...')

    text_emb = []
    for name in class_names:
        emb = encode([template.format(name) for template in templates]).float()
        emb = emb / emb.norm(dim=-1, keepdim=True)
        text_emb.append(emb.mean(dim=0))
    return torch.stack(text_emb).cpu()


def semantic_etf_anchors(num_classes, dim, lam, text_emb=None, shuffle=False, seed=0):
    '''
    Algorithm 1: T_lambda in R^{C x d}, row t_k is the fixed anchor of class k
    text_emb: C x D prompt-ensembled CLIP embeddings u_k (only needed when lam > 0)
    '''
    C = num_classes
    assert dim >= C, 'FedSem needs anchor dimension d >= C, got d={} and C={}'.format(dim, C)
    one = torch.ones(C, C, dtype=torch.float64)
    I = torch.eye(C, dtype=torch.float64)

    # simplex ETF Gram matrix
    G_etf = C / (C - 1) * (I - one / C)
    if lam > 0:
        # centre to undo the narrow cone, then the semantic Gram matrix
        u = text_emb.double()
        v = u - u.mean(dim=0, keepdim=True)
        v = v / v.norm(dim=1, keepdim=True)
        G_sem = torch.matmul(v, v.T)
    else:
        G_sem = torch.zeros_like(G_etf)
    # PSD with unit diagonal (convex combination)
    G = (1 - lam) * G_etf + lam * G_sem

    # G = Q Lambda Q^T, T = [Q Lambda^{1/2}  0] R
    evals, Q = torch.linalg.eigh(G)
    T = Q * evals.clamp(min=0).sqrt()
    T = torch.cat([T, torch.zeros(C, dim - C, dtype=torch.float64)], dim=1)
    generator = torch.Generator().manual_seed(seed)
    R, _ = torch.linalg.qr(torch.randn(dim, dim, generator=generator, dtype=torch.float64))
    T = torch.matmul(T, R)
    assert torch.allclose(torch.matmul(T, T.T), G, atol=1e-6), torch.max(torch.abs(torch.matmul(T, T.T) - G))

    if shuffle:
        # control: Pi T_lambda, identical geometry but wrong class assignment
        T = T[torch.randperm(C, generator=generator)]

    return T.float()


def build_fedsem_anchors(args, train_set, num_classes, dim):
    '''
    server side, once: clients never run CLIP, they only receive the C x d anchor matrix
    '''
    text_emb = None
    if args.fedsem_lambda > 0:
        class_names = get_class_names(train_set)
        if args.fedsem_text_emb and os.path.exists(args.fedsem_text_emb):
            cache = torch.load(args.fedsem_text_emb)
            assert cache['class_names'] == class_names, 'cached text embeddings were computed for different class names'
            text_emb = cache['text_emb']
        else:
            # keep the RNG stream identical with and without CLIP (e.g. lambda = 0 vs. lambda > 0, cache hit vs. miss)
            with torch.random.fork_rng():
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                text_emb = encode_class_names(class_names, CLIP_TEMPLATES, args.clip_backend, args.clip_model, device)
            if args.fedsem_text_emb:
                torch.save({'class_names': class_names, 'templates': CLIP_TEMPLATES, 'clip_backend': args.clip_backend,
                            'clip_model': args.clip_model, 'text_emb': text_emb}, args.fedsem_text_emb)
        assert text_emb.shape[0] == num_classes

    anchors = semantic_etf_anchors(num_classes, dim, args.fedsem_lambda, text_emb, bool(args.fedsem_shuffle), args.random_seed)

    cos = torch.matmul(anchors, anchors.T)
    off_diag = cos[~torch.eye(num_classes, dtype=torch.bool)]
    print('FedSem anchors: C={}, d={}, lambda={}, shuffled={}, anchor cosine min/mean/max = {:.3f}/{:.3f}/{:.3f}'.format(
        num_classes, dim, args.fedsem_lambda, bool(args.fedsem_shuffle), off_diag.min().item(), off_diag.mean().item(), off_diag.max().item()))

    return anchors
