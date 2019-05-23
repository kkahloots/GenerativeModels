"""AECNN.py:  CNN Autoencoder model"""
__author__      = "Khalid M. Kahloot"
__copyright__   = "Copyright 2019, Only for profesionals"

import os
import tensorflow as tf
import numpy as np
import sys
sys.path.append('..')

from .AE_BASE import AE_BASE
import _utils.utils as utils

class AECNN(AE_BASE):
    def __init__(self, *argz, **kwrds):
        super(AECNN, self).__init__(*argz, **kwrds)
        self.config.model_name = 'AECNN'
        self.config.model_type = 1 
        self.setup_logging()
        
    def _build(self):
        '''  ---------------------------------------------------------------------
                            COMPUTATION GRAPH (Build the model)
        ---------------------------------------------------------------------- '''
        from Alg_AE.AE_model import AEModel
        self.model = AEModel(self.network_params,sigma_act=utils.softplus_bias,
                               transfer_fct=tf.nn.relu, learning_rate=self.config.l_rate,
                               kinit=tf.contrib.layers.xavier_initializer(),
                               batch_size=self.config.batch_size, dropout=self.config.dropout, batch_norm=self.config.batch_norm, 
                               epochs=self.config.epochs, checkpoint_dir=self.config.checkpoint_dir, 
                               summary_dir=self.config.summary_dir, result_dir=self.config.results_dir, 
                               restore=self.flags.restore, plot=self.flags.plot, model_type=self.config.model_type)
        print('building AECNN Model...')
        print('\nNumber of trainable paramters', self.model.trainable_count)