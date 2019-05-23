# -*- coding: utf-8 -*-
AE = 0
AECNN = 1


# Stopping tolerance
tol = 1e-6

def get_model_name(model, config):
    if model=='AE' or model=='AECNN':
        return get_model_name_AE(model, config)
        
def get_model_name_AE(model, config):
    model_name = model + '_' \
                 + config.dataset_name+ '_'  \
                 + 'latent_dim' + str(config.latent_dim) + '_' \
                 + 'h_dim' + str(config.hidden_dim)  + '_' \
                 + 'h_nl' + str(config.num_layers)
    return model_name
