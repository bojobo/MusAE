from keras import backend as k
from keras.optimizers import Adam

regularisation_weight = k.variable(0)
reconstruction_weight = 1
z_lambda = 10  # weight of gradient penalty loss
aae_optim = Adam(1e-5, clipnorm=1., clipvalue=.5),  # decay=1e-4),
n_epochs = 100
test_size = 0.2
