from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np


def compute_instance_level_score_nltk(gts, res):
    """Instance level bleu score for sanity check
    by xiaodl
    """
    assert(sorted(gts.keys()) == sorted(res.keys()))
    imgIds = list(gts.keys())
    score_list = []
    smooth = SmoothingFunction()
    # smooth_func = smooth.method1
    smooth_func = smooth.method0
    for id in imgIds:
        hypo = res[id][0].split()
        ref = gts[id]
        ref = [r.split() for r in ref]

        # Sanity check.
        assert(type(hypo) is list)
        assert(len(hypo) > 0)
        assert(type(ref) is list)
        assert(len(ref) >= 1)
        bleu_1 = sentence_bleu(ref, hypo, weights=(1.0, 0.0), smoothing_function=smooth_func)
        bleu_2 = sentence_bleu(ref, hypo, weights=(0.5, 0.5), smoothing_function=smooth_func)
        bleu_3 = sentence_bleu(ref, hypo, weights=(1./3, 1./3, 1./3), smoothing_function=smooth_func)
        bleu_4 = sentence_bleu(ref, hypo, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_func)
        score_list.append([bleu_1, bleu_2, bleu_3, bleu_4])
    return np.asarray(score_list).mean(0).tolist(), score_list
