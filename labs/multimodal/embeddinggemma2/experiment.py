"""Small paired text/audio alignment pilot. All paths are relative to this lab."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Keep model caches and generated files inside this repository.
os.environ['HF_HOME'] = str(ROOT / '.cache/huggingface')
os.environ['TORCH_HOME'] = str(ROOT / '.cache/torch')

import numpy as np


def local_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f'Path must stay inside {ROOT}: {value}')
    return path


def manifest(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) < 2:
        raise ValueError('At least two paired utterances are required')
    seen = set()
    for row in rows:
        for key in ('id', 'child_id', 'session_id', 'text', 'audio_path'):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f'Missing/non-string {key}: {row}')
        if row['id'] in seen:
            raise ValueError(f'Duplicate id: {row["id"]}')
        seen.add(row['id'])
        audio = local_path(row['audio_path'])
        if not audio.is_file():
            raise ValueError(f'Audio file not found: {audio}')
        # Inputs must already be utterance clips; never silently embed a whole session.
        if 'start_ms' in row or 'end_ms' in row:
            raise ValueError('Crop utterance audio first; segment timestamps are not supported')
    return rows


def normalized(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Embeddings must be a finite 2D matrix')
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError('Zero embedding vector')
    return x / norms


def retrieval(scores):
    # Conservative tie handling: tied candidates count ahead of the correct pair.
    ranks = np.sum(scores >= np.diag(scores)[:, None], axis=1)
    return {'recall_at_1': float(np.mean(ranks <= 1)),
            'recall_at_5': float(np.mean(ranks <= 5)),
            'mrr': float(np.mean(1 / ranks))}


def alignment(text, audio, rows, seed=42):
    from sklearn.metrics import silhouette_score
    text, audio = normalized(text), normalized(audio)
    if text.shape != audio.shape or len(text) != len(rows):
        raise ValueError('Paired embedding shapes and metadata length must agree')
    n = len(rows)
    if n < 2:
        raise ValueError('At least two pairs are required')
    scores = np.clip(text @ audio.T, -1, 1)
    matched = np.diag(scores)
    # Every mismatched pair is used; avoids accidentally pairing an item with itself.
    unmatched = scores[~np.eye(n, dtype=bool)]
    vectors = np.concatenate([text, audio])
    result = {
        'n_pairs': n, 'n_children': len({r['child_id'] for r in rows}),
        'seed': seed, 'matched_cosine_mean': float(matched.mean()),
        'matched_cosine_std': float(matched.std()),
        'mismatched_cosine_mean': float(unmatched.mean()),
        'matched_minus_mismatched': float(matched.mean() - unmatched.mean()),
        'text_to_audio': retrieval(scores), 'audio_to_text': retrieval(scores.T),
        'modality_silhouette_cosine': float(silhouette_score(
            vectors, [0] * n + [1] * n, metric='cosine')),
        'topic_silhouette_cosine': None,
        'notes': ['Exact-pair retrieval; repeated equivalent transcripts can be false negatives.',
                  'Descriptive pilot metrics; no clinical or population-level inference.',
                  'Silhouettes use original vectors, not projected coordinates.'],
    }
    topics = [r.get('topic') for r in rows]
    if all(isinstance(t, str) and t.strip() for t in topics) and 1 < len(set(topics)) < n:
        result['topic_silhouette_cosine'] = float(silhouette_score(vectors, topics * 2, metric='cosine'))
    pairs = [dict(id=row['id'], child_id=row['child_id'], session_id=row['session_id'],
                  cosine=float(matched[i]), gap=float(1 - matched[i]))
             for i, row in enumerate(rows)]
    return result, pairs


def require_dialogue_verification(rows, allow_assumed=False):
    pending = [r['id'] for r in rows if r.get('analysis_unit') == 'all_speaker_dialogue'
               and r.get('manual_alignment_verified') is not True
               and not (allow_assumed and r.get('alignment_assumed_by_user') is True)]
    if pending:
        raise ValueError('Dialogue content and boundaries require listening verification before Exp 1: '
                         + ', '.join(pending))


def extract(args):
    manifest_hash = hashlib.sha256(local_path(args.manifest).read_bytes()).hexdigest()
    rows = manifest(local_path(args.manifest))
    require_dialogue_verification(rows, args.allow_assumed_alignment)
    import torch
    from sentence_transformers import SentenceTransformer
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if device.startswith('cuda') and torch.cuda.is_bf16_supported() else torch.float32
    model = SentenceTransformer(args.model, device=device,
                                config_kwargs={'vision_config': None},
                                model_kwargs={'dtype': dtype}, revision=args.revision)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    with torch.inference_mode():
        text = model.encode([r['text'] for r in rows], batch_size=args.batch_size,
                            normalize_embeddings=True, show_progress_bar=True)
        audio = model.encode([{'audio': str(local_path(r['audio_path']))} for r in rows],
                             batch_size=args.batch_size, normalize_embeddings=True,
                             show_progress_bar=True)
    text, audio = normalized(text), normalized(audio)
    if text.shape != (len(rows), 768) or audio.shape != text.shape:
        raise ValueError(f'Expected paired 768d embeddings: {text.shape}, {audio.shape}')
    out = local_path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'embeddings.npz', text=text, audio=audio)
    (out / 'rows.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    provenance = {'model': args.model, 'requested_revision': args.revision,
                  'resolved_revision': getattr(model[0].auto_model.config, '_commit_hash', None),
                  'device': device, 'dtype': str(dtype),
                  'allow_assumed_alignment': args.allow_assumed_alignment,
                  'manifest_sha256': manifest_hash,
                  'audio_sha256': {r['id']: hashlib.sha256(local_path(r['audio_path']).read_bytes()).hexdigest() for r in rows},
                  'versions': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'sentence-transformers', 'numpy')}}
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2))


def evaluate(args):
    folder = local_path(args.input)
    rows = json.loads((folder / 'rows.json').read_text())
    require_dialogue_verification(rows, args.allow_assumed_alignment)
    with np.load(folder / 'embeddings.npz', allow_pickle=False) as data:
        result, pairs = alignment(data['text'], data['audio'], rows)
    result['alignment_review'] = {
        'verified_pairs': sum(r.get('manual_alignment_verified') is True for r in rows),
        'assumed_pairs': sum(r.get('manual_alignment_verified') is not True and r.get('alignment_assumed_by_user') is True for r in rows),
        'allow_assumed_alignment': args.allow_assumed_alignment,
    }
    (folder / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    (folder / 'pair_scores.json').write_text(json.dumps(pairs, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    validate = commands.add_parser('validate')
    validate.add_argument('--manifest', required=True)
    ext = commands.add_parser('extract')
    ext.add_argument('--manifest', required=True)
    ext.add_argument('--output', default='outputs/pilot')
    ext.add_argument('--model', default='google/embeddinggemma-2')
    ext.add_argument('--revision', default='main', help='Use a commit hash for reproducible extraction')
    ext.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    ext.add_argument('--batch-size', type=int, default=4)
    ev = commands.add_parser('evaluate')
    ev.add_argument('--input', default='outputs/pilot')
    for command in (ext, ev):
        command.add_argument('--allow-assumed-alignment', action='store_true',
                             help='Exploratory run: accept pairs explicitly assumed aligned by user')
    args = parser.parse_args()
    if args.command == 'validate':
        rows = manifest(local_path(args.manifest))
        print(f'Validated {len(rows)} pairs, {len({r["child_id"] for r in rows})} children')
    elif args.command == 'extract':
        if args.batch_size < 1:
            parser.error('--batch-size must be positive')
        extract(args)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
