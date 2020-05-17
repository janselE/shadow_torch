import torch

def _acc(preds, targets, num_k):
    acc = int((preds == targets).sum()) / float(preds.shape[0])
    return acc

def original_match(flat_preds, flat_targets, preds_k, targets_k):
    out_to_gts = {}
    out_to_gts_scores = {}

    for out_c in range(preds_k):
        for gt_c in range(targets_k):
          # the amount of out_c at all the gt_c samples
            tp_score = int(((flat_preds == out_c) * (flat_targets == gt_c)).sum())

            if (out_c not in out_to_gts) or (tp_score > out_to_gts_scores[out_c]):
                out_to_gts[out_c] = gt_c
                out_to_gts_scores[out_c] = tp_score

    return list(out_to_gts.items())


def eval_acc(flat_preds, flat_targets):
    num_samples = flat_targets.shape[0]
    match = original_match(flat_preds, flat_targets, 12, 12)

    found = torch.zeros(12)
    reordered_preds = torch.zeros(num_samples, dtype=flat_preds[0].dtype) 

    for pred_i, target_i in match:
        reordered_preds[flat_preds == pred_i] = target_i
        found[pred_i] = 1

    acc = _acc(reordered_preds, flat_targets, 12)
    return acc
