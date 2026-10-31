"""NumPy evaluator for numeric symmetric trees exported from CatBoost.

The training library is unnecessary at inference. Only FloatFeature splits
and one scalar leaf value are accepted; unsupported model formats fail closed.
"""
import numpy as np


def evaluate(x, weights):
    x=np.asarray(x,dtype=np.float32)
    if x.ndim!=2 or not np.isfinite(x).all():
        raise ValueError('Expected a finite feature matrix')
    indices=weights['feature_indices'];borders=weights['borders'];depths=weights['depths']
    if indices.shape!=borders.shape or len(depths)!=len(indices):
        raise ValueError('Inconsistent tree arrays')
    result=np.zeros(len(x),dtype=np.float64)
    for tree,depth in enumerate(depths):
        depth=int(depth);ix=indices[tree,:depth]
        if np.any(ix<0) or np.any(ix>=x.shape[1]):raise ValueError('Feature index out of bounds')
        leaf=np.zeros(len(x),dtype=np.int32)
        for level,column in enumerate(ix):
            leaf|=(x[:,column]>borders[tree,level]).astype(np.int32)<<level
        result+=weights['leaf_values'][tree,leaf]
    return result*float(weights['scale'])+float(weights['bias'])


def from_json(model):
    info=model['features_info']
    if info.get('categorical_features') or info.get('ctrs') or info.get('text_features'):
        raise ValueError('Only numeric models are supported')
    features={v['feature_index']:v['flat_feature_index'] for v in info['float_features']}
    trees=model['oblivious_trees'];depths=np.asarray([len(t.get('splits') or []) for t in trees],np.int32)
    depth=int(max(depths,default=0))
    if depth>8:raise ValueError('Prototype resource bound is depth 8')
    indices=np.zeros((len(trees),depth),np.int32);borders=np.zeros((len(trees),depth),np.float32)
    leaves=np.zeros((len(trees),2**depth),np.float64)
    for i,t in enumerate(trees):
        if len(t['leaf_values'])!=2**int(depths[i]):raise ValueError('Expected scalar leaves')
        leaves[i,:len(t['leaf_values'])]=t['leaf_values']
        for j,split in enumerate(t.get('splits') or []):
            if split['split_type']!='FloatFeature':raise ValueError('Unsupported tree split')
            indices[i,j]=features[split['float_feature_index']];borders[i,j]=split['border']
    scale,bias=model['scale_and_bias']
    if len(bias)!=1:raise ValueError('Expected scalar output')
    return dict(feature_indices=indices,borders=borders,depths=depths,leaf_values=leaves,
                scale=np.asarray(scale,np.float64),bias=np.asarray(bias[0],np.float64))
