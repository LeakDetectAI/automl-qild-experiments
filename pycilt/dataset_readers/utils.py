import logging

import numpy as np

GEN_TYPES = ['single', 'multiple']


def generate_samples_per_class(n_classes, samples=1000, imbalance=0.05, gen_type='single', logger=None):
    if logger is None:
        logger = logging.getLogger("Generate Samples")
    logger.info("###############################################################################")
    if imbalance > 1 / n_classes:
        raise ValueError(f"The imbalance {imbalance} for a class cannot be more than uniform {1 / n_classes}")
    if gen_type not in GEN_TYPES:
        raise ValueError(f"Generation type {gen_type} not defined {GEN_TYPES}")
    assert (n_classes == 2) == (gen_type == 'single') or n_classes > 2
    samples_per_class = {}
    n_total_instances = samples * n_classes
    if gen_type == 'single':
        for n_c in range((n_classes - 1)):
            imb = ((1 - imbalance) / (n_classes - 1))
            n_samples = imb * n_total_instances
            samples_per_class[str(n_c)] = int(np.ceil(n_samples))
            logger.info(f"Class {n_c + 1} calculated {n_samples / n_total_instances}")
        samples_per_class[str(n_classes - 1)] = n_total_instances - sum(samples_per_class.values())
        logger.info(f"Class {n_classes} calculated {samples_per_class[str(n_classes - 1)] / n_total_instances}")
    if gen_type == 'multiple':
        for n_c in range((n_classes - 1)):
            n_samples = imbalance * n_total_instances
            samples_per_class[str(n_c)] = int(np.ceil(n_samples))
            logger.info(f"Class {n_c + 1} calculated {n_samples / n_total_instances}")
        samples_per_class[str(n_classes - 1)] = n_total_instances - sum(samples_per_class.values())
        logger.info(f"Class {n_classes} calculated {samples_per_class[str(n_classes - 1)] / n_total_instances}")
    logger.info(f"Imbalanced {imbalance} samples_per_class {samples_per_class}")
    return samples_per_class


def clean_class_label(string):
    string = ' '.join(string.split('_')).title()
    string = string.replace("  ", " ")
    return string
