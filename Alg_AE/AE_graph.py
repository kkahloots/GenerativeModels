"""AE_graph.py: Tensorflow Graph for the Autoencoder"""
__author__      = "Khalid M. Kahloot"
__copyright__   = "Copyright 2019, Only for profesionals"

from base.base_graph import BaseGraph
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from networks.dense_net import DenseNet

'''
This is the Main AEGraph.
'''
class AEGraph(BaseGraph):
    def __init__(self, network_params, sigma_act=tf.nn.softplus,
                 transfer_fct=tf.nn.relu, learning_rate=1e-4, 
                 kinit=tf.contrib.layers.xavier_initializer(), batch_size=32,
                 reuse=None, dropout=0.2):
        
        super().__init__(learning_rate)
        
        self.width = network_params['input_width']
        self.height = network_params['input_height']
        self.nchannel = network_params['input_nchannels']
        
        self.hidden_dim = network_params['hidden_dim']
        self.latent_dim = network_params.latent_dim    
        self.num_layers = network_params['num_layers'] # Num of Layers in P(x|z)
        self.l2 = network_params.l2
        self.dropout = dropout         
        
        self.sigma_act = sigma_act # Actfunc for NN modeling variance
        
        self.x_flat_dim = self.width * self.height * self.nchannel
        
        self.transfer_fct = transfer_fct
        self.kinit = kinit
        self.bias_init = tf.constant_initializer(0.0)
        self.batch_size = batch_size

        self.reuse = reuse

    def build_graph(self):
        self.create_inputs()
        self.create_graph(self.x_batch_flat)
        self.create_loss_optimizer()
    
    def create_inputs(self):
        with tf.variable_scope('inputs', reuse=self.reuse):
            self.x_batch = tf.placeholder(tf.float32, [self.batch_size, self.width, self.height, self.nchannel], name='x_batch')
            self.x_batch_flat = tf.reshape(self.x_batch , [-1,self.x_flat_dim])
            
            self.z_batch = tf.placeholder(tf.float32, [self.batch_size, self.latent_dim], name='z_batch')
    ''' 
    ------------------------------------------------------------------------------
                                     GRAPH FUNCTIONS
    ------------------------------------------------------------------------------ 
    '''    
        
    def create_graph(self, input):
        print('\n[*] Defining encoder...')
        
        with tf.variable_scope('encoder', reuse=self.reuse):
            Qlatent_x = self.create_encoder(input_=input,
                            hidden_dim=self.hidden_dim, 
                            output_dim=self.latent_dim, 
                            num_layers=self.num_layers, 
                            transfer_fct=self.transfer_fct,
                            act_out=None, 
                            reuse=self.reuse, 
                            kinit=self.kinit,
                            bias_init=self.bias_init,
                            drop_rate=self.dropout)
        
            self.latent = Qlatent_x.output
            self.z_batch = self.latent
            
        print('\n[*] Defining decoder...')
        with tf.variable_scope('decoder', reuse=self.reuse):
            Px_latent = self.create_decoder(input_=self.z_batch,
                            hidden_dim=self.hidden_dim, 
                            output_dim=self.x_flat_dim, 
                            num_layers=self.num_layers, 
                            transfer_fct=self.transfer_fct,
                            act_out=tf.nn.sigmoid, 
                            reuse=self.reuse, 
                            kinit=self.kinit,
                            bias_init=self.bias_init,
                            drop_rate=self.dropout)
        
            self.x_recons_flat = Px_latent.output
        self.x_recons = tf.reshape(self.x_recons_flat , [-1,self.width, self.height, self.nchannel])
    '''  
    ------------------------------------------------------------------------------
                                     ENCODER-DECODER
    ------------------------------------------------------------------------------ 
    '''          
    def create_encoder(self, input_,hidden_dim,output_dim,num_layers,transfer_fct, \
                       act_out,reuse,kinit,bias_init, drop_rate):
        latent_ = DenseNet(input_=input_,
                            hidden_dim=hidden_dim, 
                            output_dim=output_dim, 
                            num_layers=num_layers, 
                            transfer_fct=transfer_fct,
                            act_out=act_out, 
                            reuse=reuse, 
                            kinit=kinit,
                            bias_init=bias_init,
                            drop_rate=drop_rate)        
        return latent_
    
    def create_decoder(self,input_,hidden_dim,output_dim,num_layers,transfer_fct, \
                       act_out,reuse,kinit,bias_init, drop_rate):
        recons_ = DenseNet(input_=input_,
                            hidden_dim=hidden_dim, 
                            output_dim=output_dim, 
                            num_layers=num_layers, 
                            transfer_fct=transfer_fct,
                            act_out=act_out, 
                            reuse=reuse, 
                            kinit=kinit,
                            bias_init=bias_init,
                            drop_rate=drop_rate)        
        return recons_

    '''  
    ------------------------------------------------------------------------------
                                     LOSSES
    ------------------------------------------------------------------------------ 
    '''    
    
    def create_loss_optimizer(self):
        print('[*] Defining Loss Functions and Optimizer...')
        with tf.name_scope('reconstruct'):
            
            self.reconstruction = self.get_ell(self.x_batch_flat, self.x_recons_flat)
            
        self.loss_reconstruction_m = tf.reduce_mean(self.reconstruction)

        with tf.variable_scope("L2_loss", reuse=self.reuse):
            tv = tf.trainable_variables()
            self.regularization_cost = tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ])
        
        with tf.variable_scope("ae_loss", reuse=self.reuse):
            self.ae_loss = tf.reduce_mean(self.reconstruction)  + self.l2*self.regularization_cost # shape = [None,]

        with tf.variable_scope("optimizer" ,reuse=self.reuse):
            self.optimizer = tf .train.AdamOptimizer(self.learning_rate)
            self.train_step = self.optimizer.minimize(self.ae_loss, global_step=self.global_step_tensor)

    ## ------------------- LOSS: EXPECTED LOWER BOUND ----------------------
    def get_ell(self, x, x_recons):
        """
        Returns the expected log-likelihood of the lower bound.
        For this we use a bernouilli LL.
        """
        # p(x|z)        
        return - tf.reduce_sum((x) * tf.log(x_recons + 1e-10) +
                               (1 - x) * tf.log(1 - x_recons + 1e-10), 1)
                               
    '''  
    ------------------------------------------------------------------------------
                                     FIT & EVALUATE TENSORS
    ------------------------------------------------------------------------------ 
    '''
    
    def partial_fit(self, session, x):
        tensors = [self.train_step, self.ae_loss, self.loss_reconstruction_m, self.regularization_cost]
        feed_dict = {self.x_batch: x}
        _, loss, recons, L2_loss  = session.run(tensors, feed_dict=feed_dict)
        return loss, recons, L2_loss
    
    def evaluate(self, session, x):
        tensors = [self.ae_loss, self.loss_reconstruction_m, self.regularization_cost]
        feed_dict = {self.x_batch: x}
        loss, recons, L2_loss  = session.run(tensors, feed_dict=feed_dict)
        return loss, recons, L2_loss

    '''  
    ------------------------------------------------------------------------------
                                     GENERATE LATENT and RECONSTRUCT
    ------------------------------------------------------------------------------ 
    '''
        
    def reconst_loss(self, session, x):
        tensors= [self.reconstruction] 
        feed = {self.x_batch: x}  
        return session.run(tensors, feed_dict=feed) 
     
        
    '''  
    ------------------------------------------------------------------------------
                                         GRAPH OPERATIONS
    ------------------------------------------------------------------------------ 
    '''    
              
    def encode(self, session, inputs):
        tensors = [self.latent]
        feed_dict = {self.x_batch: inputs}
        return session.run(tensors, feed_dict=feed_dict)
        
    def decode(self, session, z):
        tensors = [self.x_recons]        
        feed_dict = {self.latent: z}
        return session.run(tensors, feed_dict=feed_dict) 
    
