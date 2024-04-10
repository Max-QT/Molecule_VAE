import keras
import pandas as pd
import sys
sys.path.append("D:/Code/chemical_vae-main/chemvae")
from chemvae.train_vae_2 import load_models
from chemvae import hyperparameters

params = hyperparameters.load_params("models/train_5/exp.json")
AE_only_model, AE_PP_model, encoder, decoder, property_predictor, kl_loss_var = load_models(params)
keras.utils.plot_model(AE_PP_model, "AE_PP_model.png", show_shapes=True)  # 画网络结构图
keras.utils.plot_model(encoder, "encoder.png", show_shapes=True)  # 画网络结构图
keras.utils.plot_model(decoder, "decoder.png", show_shapes=True)  # 画网络结构图
keras.utils.plot_model(property_predictor, "property_predictor.png", show_shapes=True)  # 画网络结构图
