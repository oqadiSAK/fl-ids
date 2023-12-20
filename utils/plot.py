import data_loader as data_loader
import model_loader as model_loader
from keras.utils import plot_model

X_train, _, _, _ = data_loader.get_data()
model = model_loader.get_model(X_train.shape[1:])
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)