from functools import partial
from pprint import pprint

import click
import fasttext
import numpy as np
import pandas as pd
import sklearn
from scipy import stats


@click.command()
@click.option('--train', prompt='train CSV file path', help='train CSV file path.')
@click.option('--test', prompt='test CSV file path', help='test CSV file path.')
@click.option('--topics_column', default='labels', prompt='Topics Column name', help='The name of topics column.')
@click.option('--readme_column', default='text', prompt='Text Column name', help='The name of readme text column.')
@click.option('--model_output', default='hmc_lmlp_model', help='Model save path.')
@click.option('--learning_rate', default=0.05, help='Learning rate Value.')
@click.option('--epoch', default=100, help='Number of Epoch.')
@click.option('--word_ngrams', default=2, help='Number of wordNgrams.')
def ft(train, test, topics_column, readme_column, model_output, learning_rate, epoch, word_ngrams):
    
    return 0