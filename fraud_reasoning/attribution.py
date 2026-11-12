"""Exact additive tree-path decomposition of RAW RF probabilities, not SHAP or causal reasoning."""
import numpy as np

def explain_forest(model, imputer, frame, feature_names, top_k=8):
    vec = np.asarray(imputer.transform(frame), dtype=float)[0]
    classes = list(model.classes_)
    positive = classes.index(1)
    contributions = np.zeros(len(feature_names), dtype=float)
    root_sum = 0.0
    for tree in model.estimators_:
        t = tree.tree_
        def prob(node):
            dist = np.asarray(t.value[node][0], dtype=float)
            return float(dist[positive] / max(dist.sum(), 1e-12))
        node=0
        root_sum += prob(0)
        while t.feature[node] >= 0:
            f=int(t.feature[node])
            next_node = int(t.children_left[node] if vec[f] <= t.threshold[node] else t.children_right[node])
            contributions[f] += prob(next_node) - prob(node)
            node=next_node
    n=len(model.estimators_)
    contributions/=n
    base=root_sum/n
    predicted=float(model.predict_proba(np.asarray([vec]))[0,positive])
    reconstruction=base+float(contributions.sum())
    if abs(reconstruction-predicted)>1e-5:
        raise ArithmeticError(f'Tree-path attribution not additive: {reconstruction} != {predicted}')
    ordering=np.argsort(-np.abs(contributions))[:top_k]
    explanation=[]
    for i in ordering:
        val=float(contributions[i]); f=feature_names[i]
        explanation.append({'feature':f,'observed_value':None if not np.isfinite(vec[i]) else float(vec[i]),
                            'direction':'toward_fraud' if val>0 else ('away_from_fraud' if val<0 else 'neutral'),
                            'raw_probability_contribution':round(val,8),
                            'explanation':'Local tree-path attribution to raw RF output, not a causal explanation or contribution to calibrated probability.'})
    return {'method':'averaged_tree_path_probability_differences', 'raw_fraud_probability':predicted,
            'raw_probability_baseline':base, 'reconstruction_error':abs(reconstruction-predicted),
            'attributions_are_for_calibrated_probability':False,
            'all_feature_contributions': {feature_names[i]: float(contributions[i]) for i in range(len(feature_names))},
            'reasoning':explanation}
