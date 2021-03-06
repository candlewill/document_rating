__author__ = 'NLP-PC'
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import numpy as np
import math
from log_manager import log_performance, log_state

def evaluate(true, pred, msg):
    true, pred = np.array(true), np.array(pred)
    MAE= mean_absolute_error(np.array(true), np.array(pred))
    MSE_sqrt = math.sqrt(mean_squared_error(np.array(true), np.array(pred)))
    MSE = mean_squared_error(np.array(true), np.array(pred))
    R2 = r2_score(np.array(true), np.array(pred))
    Pearson_r = pearsonr(np.array(true), np.array(pred))
    Spearman_r = spearmanr(np.array(true), np.array(pred))
    log_state(msg)
    log_performance(MSE, MAE, Pearson_r, R2, Spearman_r, MSE_sqrt)
    return None