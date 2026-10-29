"""Expanded Exp 1 analysis. See EXP1_PROTOCOL.md for frozen choices and limitations."""
import json
import hashlib
import importlib.metadata
import os
from collections import Counter
from pathlib import Path
import numpy as np
from experiment import ROOT, normalized, require_dialogue_verification, alignment, retrieval

os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'.cache/numba')
TOPICS={
 'book_story': {'book','books','story','stories','read','reading','page','pages'},
 'vehicles': {'car','cars','truck','trucks','drive','driving','wheel','wheels','trunk','garage','motorcycle','bus'},
 'animals': {'cat','cats','kitty','dog','dogs','hippo','elephant','lion','tiger','bird','birds','fish','rabbit','bunny'},
 'food': {'eat','eating','food','lunch','dinner','breakfast','cookie','cookies','milk','sandwich','hungry'},
 'health': {'medicine','tummy','sick','hurt','hurts','doctor','hospital'},
 'music': {'piano','music','sing','singing','song'},
}


def ranks(scores):
    return (scores>=np.diag(scores)[:,None]).sum(axis=1)


def grouped(scores, labels):
    groups={}
    for key in sorted(set(labels)):
        indices=np.flatnonzero(np.asarray(labels)==key)
        block=scores[np.ix_(indices,indices)]
        groups[key]={'n_candidates':len(indices),'text_to_audio':retrieval(block),'audio_to_text':retrieval(block.T)}
    return groups


def weak_topic(text):
    import re
    words=set(re.findall(r'[a-z]+',text.lower()))
    matches=[topic for topic,keys in TOPICS.items() if words & keys]
    return matches[0] if len(matches)==1 else 'mixed' if matches else 'unassigned'


def per_query(scores, rows):
    diag=np.diag(scores)
    negative=scores.copy(); np.fill_diagonal(negative,-np.inf)
    tr, ar=ranks(scores),ranks(scores.T)
    result=[]
    for i,row in enumerate(rows):
        mask=np.asarray([r['session_id']==row['session_id'] for r in rows]);mask[i]=False
        result.append(dict(id=row['id'],child_id=row['child_id'],session_id=row['session_id'],
                           weak_topic=weak_topic(row['text']),cosine=float(diag[i]),
                           same_session_negative_mean=float(scores[i,mask].mean()),
                           text_rank=int(tr[i]),audio_rank=int(ar[i]),
                           text_margin=float(diag[i]-negative[i].max()),audio_margin=float(diag[i]-negative[:,i].max()),
                           text_top_audio=rows[int(scores[i].argmax())]['id'],
                           audio_top_text=rows[int(scores[:,i].argmax())]['id']))
    return result


def uncertainty(scores, rows, per, seed=20261028, bootstrap=2000, permutations=1000):
    rng=np.random.default_rng(seed)
    children=sorted({r['child_id'] for r in rows})
    child_indices=[np.flatnonzero(np.asarray([r['child_id'] for r in rows])==c) for c in children]
    per_child={}
    for child,ix in zip(children,child_indices):
        per_child[child]={
            'n_pairs':len(ix),
            'matched_minus_same_session_negative':float(np.mean([per[i]['cosine']-per[i]['same_session_negative_mean'] for i in ix])),
            'global_text_recall_at_1':float(np.mean([per[i]['text_rank']==1 for i in ix])),
            'global_audio_recall_at_1':float(np.mean([per[i]['audio_rank']==1 for i in ix])),
        }
    keys=['matched_minus_same_session_negative','global_text_recall_at_1','global_audio_recall_at_1']
    result={}
    for key in keys:
        values=np.array([per_child[c][key] for c in children])
        means=values[rng.integers(len(children),size=(bootstrap,len(children)))].mean(axis=1)
        result[key]={'child_macro_mean':float(values.mean()),'child_bootstrap_95ci':np.quantile(means,[.025,.975]).tolist()}
    sessions=[np.flatnonzero(np.asarray([r['session_id'] for r in rows])==s) for s in sorted({r['session_id'] for r in rows})]
    observed=result[keys[0]]['child_macro_mean']
    null=[]
    n=len(rows)
    for _ in range(permutations):
        mapping=np.arange(n)
        for ix in sessions: mapping[ix]=rng.permutation(ix)
        shuffled=scores[:,mapping]
        delta=np.zeros(n)
        for ix in sessions:
            b=shuffled[np.ix_(ix,ix)]
            delta[ix]=np.diag(b)-(b.sum(axis=1)-np.diag(b))/(len(ix)-1)
        null.append(float(np.mean([delta[ix].mean() for ix in child_indices])))
    result['within_session_permutation']={'iterations':permutations,'observed_child_macro_delta':observed,
        'one_sided_p':float((1+np.sum(np.array(null)>=observed))/(permutations+1)),
        'null_mean':float(np.mean(null)),'caveat':'Approximate exchangeability; temporal dependence not removed.'}
    return per_child,result


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
    from sklearn.cluster import HDBSCAN
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
    import umap
    out=ROOT/'outputs/exp1_full'
    rows=json.loads((out/'rows.json').read_text())
    require_dialogue_verification(rows,allow_assumed=True)
    with np.load(out/'embeddings.npz') as d:t,a=normalized(d['text']),normalized(d['audio'])
    result,pairs=alignment(t,a,rows)
    s=np.clip(t@a.T,-1,1)
    per=per_query(s,rows)
    child_metrics,uncertain=uncertainty(s,rows,per)
    result.update(alignment_review={'verified':sum(r['manual_alignment_verified'] for r in rows),
                                   'assumed':sum(not r['manual_alignment_verified'] for r in rows)},
                  per_child=child_metrics,uncertainty=uncertain,
                  same_session_retrieval=grouped(s,[r['session_id'] for r in rows]),
                  same_child_retrieval=grouped(s,[r['child_id'] for r in rows]))
    x=np.concatenate([t,a]);n=len(rows)
    modality=np.repeat([0,1],n)
    child=[r['child_id'] for r in rows]*2
    session=[r['session_id'] for r in rows]*2
    result['child_silhouette_cosine']=float(silhouette_score(x,child,metric='cosine'))
    result['session_silhouette_cosine']=float(silhouette_score(x,session,metric='cosine'))
    cv=cross_val_score(LogisticRegression(C=1,max_iter=2000),x,modality,
                       cv=LeaveOneGroupOut(),groups=child,scoring='balanced_accuracy')
    result['modality_probe_leave_child_out']={'balanced_accuracy_by_fold':cv.tolist(),'mean':float(cv.mean())}
    topics=np.array([weak_topic(r['text']) for r in rows])
    labelled=~np.isin(topics,['mixed','unassigned'])
    ti=np.flatnonzero(np.r_[labelled,labelled])
    result['weak_topic_analysis']={'kind':'Fixed transcript keyword rules; NOT independent gold labels',
                                  'counts':dict(Counter(topics.tolist())),
                                  'silhouette_cosine':None,'same_topic_retrieval':grouped(s,topics.tolist())}
    if len(ti)>2 and 1<len(set(topics[labelled]))<len(ti):
        result['weak_topic_analysis']['silhouette_cosine']=float(silhouette_score(x[ti],np.r_[topics,topics][ti],metric='cosine'))
    result['same_session_same_weak_topic_retrieval'] = grouped(s,[r['session_id']+'::'+topic for r,topic in zip(rows,topics)])
    (out/'weak_topic_rules.json').write_text(json.dumps({k:sorted(v) for k,v in TOPICS.items()},indent=2))
    cluster=HDBSCAN(min_cluster_size=10,min_samples=5,metric='euclidean').fit_predict(x)
    keep=cluster>=0
    nclusters=len(set(cluster[keep]))
    result['hdbscan']={'space':'original normalized 768d; Euclidean distance',
                       'min_cluster_size':10,'min_samples':5,'n_clusters':nclusters,
                       'noise_fraction':float(np.mean(~keep)),
                       'cluster_silhouette_cosine':float(silhouette_score(x[keep],cluster[keep],metric='cosine')) if 1<nclusters<keep.sum() else None,
                       'modality_ari_including_noise':float(adjusted_rand_score(modality,cluster)),
                       'weak_topic_nmi_including_noise':float(normalized_mutual_info_score(np.r_[topics,topics],cluster))}
    rng=np.random.default_rng(20261028)
    samples=[]
    for _ in range(500):
        ix=rng.choice(n,12,replace=False);b=s[np.ix_(ix,ix)]
        samples.append([retrieval(b)['recall_at_1'],retrieval(b.T)['recall_at_1']])
    result['fixed_12_candidate_subsets']={'iterations':500,'text_recall1_mean':float(np.mean(samples,axis=0)[0]),
                                        'audio_recall1_mean':float(np.mean(samples,axis=0)[1]),
                                        'note':'Descriptive repeated subsets; not independent experiments or confidence intervals.'}
    for seed in [42,7,123]:
        y=umap.UMAP(n_neighbors=15,min_dist=.1,metric='cosine',random_state=seed).fit_transform(x)
        np.save(out/f'umap_seed{seed}.npy',y)
        if seed!=42:continue
        fig,axes=plt.subplots(1,3,figsize=(16,4.5))
        for m,name,mark in [(0,'Text','o'),(1,'Audio','^')]:
            ix=modality==m;axes[0].scatter(*y[ix].T,s=15,alpha=.65,marker=mark,label=name)
        axes[0].legend();axes[0].set_title('Modality')
        for c in sorted(set(child)):
            ix=np.asarray(child)==c;axes[1].scatter(*y[ix].T,s=12,alpha=.65,label=c.replace('gleason-',''))
        axes[1].legend(fontsize=7);axes[1].set_title('Child')
        for topic in sorted(set(topics)):
            ix=np.r_[topics,topics]==topic;axes[2].scatter(*y[ix].T,s=12,alpha=.65,label=topic)
        axes[2].legend(fontsize=6);axes[2].set_title('Keyword topics (weak labels)')
        fig.suptitle('UMAP: exploratory projection; metrics computed in original 768d space')
        fig.tight_layout();fig.savefig(out/'umap.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].hist(s[~np.eye(n,dtype=bool)],bins=35,density=True,alpha=.55,label='Mismatched')
    axes[0].hist(np.diag(s),bins=20,density=True,alpha=.65,label='Matched')
    axes[0].set_xlabel('Cosine similarity');axes[0].set_ylabel('Density');axes[0].legend()
    axes[1].hist([r['text_margin'] for r in per],bins=25,alpha=.55,label='Text to audio')
    axes[1].hist([r['audio_margin'] for r in per],bins=25,alpha=.55,label='Audio to text')
    axes[1].axvline(0,color='black',linewidth=1);axes[1].set_xlabel('Matched minus strongest wrong candidate');axes[1].legend()
    fig.tight_layout();fig.savefig(out/'distributions.png',dpi=180);plt.close(fig)
    details={r['id']:r for r in rows}
    failures=[]
    for q in sorted(per,key=lambda r:r['text_margin'])[:15]:
        failures.append(dict(q,text=details[q['id']]['text'],top_audio_transcript=details[q['text_top_audio']]['text']))
    result['umap']={'seeds':[42,7,123],'n_neighbors':15,'min_dist':.1,'metric':'cosine'}
    (out/'analysis_provenance.json').write_text(json.dumps({
        'versions':{name:importlib.metadata.version(name) for name in ['numpy','scikit-learn','umap-learn','matplotlib']},
        'protocol_sha256':hashlib.sha256((ROOT/'EXP1_PROTOCOL.md').read_bytes()).hexdigest(),
        'analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'preparation_script_sha256':hashlib.sha256((ROOT/'prepare_exp1_full.py').read_bytes()).hexdigest(),
        'rows_sha256':hashlib.sha256((out/'rows.json').read_bytes()).hexdigest(),
    },indent=2))
    (out/'full_metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    (out/'query_diagnostics.json').write_text(json.dumps(per,indent=2))
    (out/'failure_examples.json').write_text(json.dumps(failures,indent=2))
    (out/'cluster_labels.json').write_text(json.dumps(cluster.tolist()))
    print(json.dumps({k:v for k,v in result.items() if k not in ['per_child','same_session_retrieval','same_child_retrieval','weak_topic_analysis']},indent=2))


if __name__=='__main__':
    main()
