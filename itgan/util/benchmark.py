import logging
import os, sys
import types
from datetime import datetime
import pandas as pd
from .data import load_dataset
from .evaluate import compute_scores
from .evaluate_cluster import compute_cluster_scores
from .base import BaseSynthesizer
import warnings
import pickle
warnings.filterwarnings(action='ignore')



def train(synthesizer, syn_arg, dataset):
    """
    Train the model and compute last scores of Clustering and scores of Supervised Learning
    """
    synthesizer = synthesizer(**syn_arg)
    train, test, meta, categoricals, ordinals = load_dataset(dataset, benchmark=True)
    synthesizer.fit(train, test, meta, dataset, categoricals, ordinals)
    synthesized = synthesizer.sample(train.shape[0])
    scores = compute_scores(train, test, synthesized, meta)
    if 'likelihood' in meta["problem_type"]:
        return scores
    scores_cluster = compute_cluster_scores(train, test, synthesized, meta)
    return scores, scores_cluster, synthesizer, synthesized