"""AE_BASE.py:  Autoencoder argument process"""
__author__      = "Khalid M. Kahloot"
__copyright__   = "Copyright 2019, Only for profesionals"

import os
import tensorflow as tf
import numpy as np
import sys
sys.path.append('..')

import _utils.utils as utils
import _utils.constants as const

class AE_BASE():
    '''  ------------------------------------------------------------------------------
                                         SET ARGUMENTS
        ---------------------------------------------------------------------------------- '''
    
    def __init__(self, dataset_name, alpha=1, beta=0.6, gamma=1, sigma=0.001, l2=1e-5, \
                       latent_dim=10, hidden_dim=100, num_layers=3,epochs=100, batch_size=32,\
                       dropout=0.2, batch_norm=True, l_rate=1e-05, restore=False, plot=False):
        args=dict()
        args['model_type']=0
        args['model_name']='AE'
        args['dataset_name']=dataset_name
        
        args['alpha']=alpha
        args['beta']=beta
        args['gamma']=gamma
        args['sigma']=sigma
        args['l2']=l2
        
        args['latent_dim']=latent_dim

        args['hidden_dim']=hidden_dim
        args['num_layers']=num_layers
        args['epochs']=epochs
        args['batch_size']=batch_size
        args['dropout']=dropout
        args['batch_norm']=batch_norm
        args['l_rate']=l_rate
        args['train']=True if restore==False else False
  
        args['plot']=plot
        args['restore']=restore
        args['early_stopping']=1
        
        dirs = ['checkpoint_dir','summary_dir', 'result_dir', 'log_dir' ]
        for d in dirs:
            args[d] = d
        
        self.config = utils.Config(args)
        
    def setup_logging(self):        
        experiments_root_dir = 'experiments'
        self.config.model_name = const.get_model_name(self.config.model_name, self.config)
        self.config.summary_dir = os.path.join(experiments_root_dir+"\\"+self.config.log_dir+"\\", self.config.model_name)
        self.config.checkpoint_dir = os.path.join(experiments_root_dir+"\\"+self.config.checkpoint_dir+"\\", self.config.model_name)
        self.config.results_dir = os.path.join(experiments_root_dir+"\\"+self.config.result_dir+"\\", self.config.model_name)

        #Flags
        flags_list = ['train', 'restore', 'plot', 'early_stopping']
        self.flags = utils.Config({ your_key: self.config.__dict__[your_key] for your_key in flags_list})
        
        # create the experiments dirs
        utils.create_dirs([self.config.summary_dir, self.config.checkpoint_dir, self.config.results_dir])
        utils.save_args(self.config.__dict__, self.config.summary_dir)
 
    def fit(self, X, y=None):
        '''  ------------------------------------------------------------------------------
                                         DATA PROCESSING
        ------------------------------------------------------------------------------ '''
            
        print('\n Processing data...')
        self.data_train, self.data_valid = utils.process_data(X, y) 
        
        print('\n building a model...')
        self.build()
        
        '''  -------------------------------------------------------------------------------
                        TRAIN THE MODEL
        ------------------------------------------------------------------------------------- '''

        if(self.flags.train):
            print('\n training a model...')
            self.model.train(self.data_train, self.data_valid, enable_es=self.flags.early_stopping)

        
    def build(self):
        '''  ------------------------------------------------------------------------------
                                     SET NETWORK PARAMS
        ------------------------------------------------------------------------------ '''        
        network_params_dict = dict()
        network_params_dict['input_height'] = self.data_train.height
        network_params_dict['input_width'] = self.data_train.width
        network_params_dict['input_nchannels'] = self.data_train.num_channels
        network_params_dict['train_size'] = self.data_train._ndata
        
        network_params_dict['hidden_dim'] =  self.config.hidden_dim
        network_params_dict['latent_dim'] =  self.config.latent_dim
        network_params_dict['l2'] =  self.config.l2

        network_params_dict['num_layers'] =  self.config.num_layers
        
        self.network_params = utils.Config(network_params_dict)  
        
        self._build()    
     
    def _build(self):
        pass

