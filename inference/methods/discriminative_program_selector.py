"""Score all four candidates with optional, quantity-bound program representations.

Programs are parsed from the supplied candidates. This model neither generates
programs nor treats missing programs as incorrect answers.
"""
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from torch import nn
from program_sequence_selector import OUTPUT, OUTPUT_ID, input_rows, vocabulary, question_batch
from quantity_features import extract
from quantity_shared_selector import _scores as base_scores
from round2_features import raw_dense
from selector import select_choices


def raw_features(bank, embeddings):
    raw = raw_dense(bank, embeddings, {'dense': 'v1_count'})
    quantities = extract(bank)
    return np.concatenate((raw[:, :, :1025], quantities['full']), -1), raw, quantities


def statistics(raw, fit):
    flat = raw[np.asarray(fit, int)].reshape(-1, raw.shape[-1]).astype(np.float64)
    return flat.mean(0).astype(np.float32), np.maximum(flat.std(0), .05).astype(np.float32)


class DiscriminativeProgram(nn.Module):
    def __init__(self, nwords, nfeatures, program_mode):
        super().__init__()
        assert program_mode in ('off', 'bound_sequence')
        self.program_mode = program_mode
        self.words = nn.Embedding(nwords, 32, padding_idx=0)
        self.question_encoder = nn.GRU(32, 32, batch_first=True, bidirectional=True)
        self.condition = nn.Linear(1024, 64, bias=False)
        self.program_tokens = nn.Embedding(len(OUTPUT), 48, padding_idx=0)
        self.quantity_binding = nn.Linear(64, 48, bias=False)
        self.program_encoder = nn.GRU(48, 48, batch_first=True)
        self.question_projection = nn.Linear(64, 48)
        self.candidate_projection = nn.Linear(nfeatures, 32)
        self.compatibility = nn.Linear(48 * 4 + 1, 32)
        self.out = nn.Linear(32, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.dropout = nn.Dropout(.2)
        self.register_buffer('feature_mean', torch.zeros(nfeatures))
        self.register_buffer('feature_scale', torch.ones(nfeatures))

    def forward(self, rows, vocab, raw):
        device = raw.device
        x = ((raw - self.feature_mean) / self.feature_scale).clamp(-5, 5)
        q, lengths, slots = question_batch(rows, vocab, device)
        packed = nn.utils.rnn.pack_padded_sequence(
            self.dropout(self.words(q)), lengths.cpu(), batch_first=True, enforce_sorted=False)
        encoded, _ = self.question_encoder(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True, total_length=q.shape[1])
        mask = q != 0
        # The supplied mean contains the question and four solutions, not a pure question vector.
        condition = torch.tanh(self.condition(x[:, :, :1024].mean(1)))
        encoded = encoded + condition[:, None] * mask[:, :, None]
        pooled = (encoded * mask[:, :, None]).sum(1) / mask.sum(1, keepdim=True)
        query = torch.tanh(self.question_projection(pooled))
        programs, owners, groups, seen = [], [], [], {}
        for i, row in enumerate(rows):
            candidate_groups = []
            for aliases in row['candidate_prefixes']:
                indices = []
                for alias in aliases:
                    key = (i, tuple(alias))
                    if key not in seen:
                        seen[key] = len(programs)
                        programs.append(alias)
                        owners.append(i)
                    indices.append(seen[key])
                candidate_groups.append(indices)
            groups.append(candidate_groups)
        zero = encoded.sum() * 0
        if programs:
            owner = torch.tensor(owners, dtype=torch.long, device=device)
            plen = torch.tensor([len(p) for p in programs], dtype=torch.long)
            tokens = torch.zeros((len(programs), int(plen.max())), dtype=torch.long, device=device)
            for i, p in enumerate(programs):
                tokens[i, :len(p)] = torch.tensor([OUTPUT_ID[t] for t in p], device=device)
            values = self.program_tokens(tokens)
            position = (tokens - OUTPUT_ID['Q0']).clamp(0, 11)
            token_slots = slots[owner].gather(1, position)
            slot_vectors = encoded[owner].gather(1, token_slots.clamp(min=0)[:, :, None].expand(-1, -1, 64))
            bound = ((tokens >= OUTPUT_ID['Q0']) & (tokens <= OUTPUT_ID['Q11']) & (token_slots >= 0))
            values = values + self.quantity_binding(slot_vectors) * bound[:, :, None]
            pp = nn.utils.rnn.pack_padded_sequence(self.dropout(values), plen, batch_first=True, enforce_sorted=False)
            _, final = self.program_encoder(pp, query[owner][None])
            representations = final[0]
        empty = torch.zeros(48, device=device) + zero
        program = torch.stack([torch.stack([representations[ix].mean(0) if ix else empty for ix in cs]) for cs in groups])
        support = torch.tensor([[bool(ix) for ix in cs] for cs in groups], device=device, dtype=raw.dtype)
        if self.program_mode == 'off':
            program = program * 0
            support = support * 0
        query = query[:, None].expand(-1, 4, -1)
        joint = torch.cat((program, query, program * query, (program - query).abs(), support[:, :, None]), -1)
        hidden = torch.tanh(self.candidate_projection(x) + self.compatibility(joint))
        # No supported-only centering, normalization, or hard correction amplitude cap.
        return self.out(self.dropout(hidden)).squeeze(-1)


def export(model, vocab, bundle, baseline=None, extra=None):
    root = Path(bundle)
    root.mkdir(parents=True, exist_ok=False)
    cfg = dict(format='mp2-discriminative-program-v1', vocabulary=vocab,
               output_tokens=OUTPUT, program_mode=model.program_mode,
               feature_dimensions=len(model.feature_mean), synthetic_zero_baseline=baseline is None)
    cfg.update(extra or {})
    if baseline is not None:
        shutil.copytree(baseline, root / 'baseline')
    (root / 'config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + '\n')
    np.savez(root / 'weights.npz', **{k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()})


def load(root, device='cpu'):
    root = Path(root)
    cfg = json.loads((root / 'config.json').read_text())
    assert cfg['format'] == 'mp2-discriminative-program-v1' and cfg['output_tokens'] == OUTPUT
    model = DiscriminativeProgram(len(cfg['vocabulary']), cfg['feature_dimensions'], cfg['program_mode']).to(device)
    with np.load(root / 'weights.npz', allow_pickle=False) as z:
        model.load_state_dict({k: torch.tensor(z[k].copy(), device=device) for k in z.files})
    model.eval()
    return model, cfg


def score_rows(model, vocab, rows, features, device='cpu'):
    result = []
    with torch.inference_mode():
        for start in range(0, len(rows), 64):
            x = torch.as_tensor(features[start:start+64], device=device)
            result.append(model(rows[start:start+64], vocab, x).cpu().numpy())
    return np.concatenate(result) if result else np.empty((0, 4), np.float32)


def candidate_scores(bank, arrays, checkpoint):
    if not bank:
        return np.empty((0, 4), np.float32)
    torch.set_num_threads(1)
    root = Path(checkpoint).resolve()
    cfg = json.loads((root / 'config.json').read_text())
    rows = input_rows(bank)
    features, raw, quantities = raw_features(bank, arrays['embeddings'])
    results = []
    for name in cfg.get('members', ['.']):
        child = (root / name).resolve()
        assert child == root or child.is_relative_to(root)
        model, c = load(child)
        baseline = np.zeros((len(bank), 4), np.float32) if c['synthetic_zero_baseline'] else base_scores(child / 'baseline', raw, quantities)
        results.append(baseline + score_rows(model, c['vocabulary'], rows, features))
    return np.mean(np.stack(results), 0).astype(np.float32)


def predict(bank, arrays, checkpoint):
    return select_choices(bank, candidate_scores(bank, arrays, checkpoint), 'max') if bank else []
