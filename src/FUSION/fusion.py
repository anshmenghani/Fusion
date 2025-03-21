#ॐ

"""
Physics is all about approximation. No one formula can account for every factor and predict 
any event with full accuracy. Over the years, scientists have used different methods to make 
these predictions, such as mathematical formulas and statistical models. This has been 
particularly difficult in fields such as astrophysics, where the subjects being studied are 
mind-blowing distances away from Earth. In recent years, the advancements in machine learning, 
specifically in neural networks, allow us to computationally model far more complex relationships 
than before. The Fusion Stellar Model is designed to model, predict, and classify stellar 
parameters more accurately than traditional methods. It takes inputs of a star’s Effective 
Temperature, Luminosity, and Radius, and runs its Physics-Informed Neural Network (PINN) algorithm 
to find twenty-three other parameters of a star such as Mass and surface Gravity. It was trained on 
the Gaia mission dataset with over 403 million stellar parameters. The performance of this model 
varies depending on the star type and parameter it predicts but has an average accuracy of 90.018% 
and an average speed of 0.008 seconds per round of predictions. While this model is currently 
limited to stars on the Hertzsprung-Russell diagram (i.e., the model cannot make accurate predictions 
on other objects in the universe, such as a neutron star), its main use is to better model the distant 
lights in our universe.  
"""

import os
os.environ["KERAS_BACKEND"] = "tensorflow" # Set the Keras backend environmental variable to Tensorflow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" # Turn off the Tesorflow OneDNN option 

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import joblib
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Layer, Input, Embedding, Flatten, LayerNormalization, Discretization, Dense, GaussianDropout, concatenate, PReLU, Softmax, Cropping1D, Reshape
from tensorflow.keras import backend as K
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.models import Model
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.losses import MeanSquaredLogarithmicError as MSLE, MeanAbsoluteError as MAE, CategoricalCrossentropy as CCE, Loss
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import gelu
from tensorflow.keras.ops import log10, tanh
from tensorflow import TensorShape, constant, cast, clip_by_value, convert_to_tensor, Variable, float32
import tensorflow.keras.callbacks as callbacks


# Set initial global variables
gen_inputs = None
submodelCount = 0
prev_val_loss = Variable(0, dtype=float32)
curr_val_loss = Variable(1, dtype=float32)


def data_prep(df, inputs, outputs, mod_attrs, func_attrs, funcs):
   """
   This function prepares the Gaia DR3 data to be used for training. More specifically, it splits the 
   data into training (and validation) and testing data. It then normalizes the data using scikit-learns 
   RobustScaler, which uses the median and interquartile range to normalize the data. This ensures 
   outliers in data do not skew the dataset too much. Next, the function applies any 
   modifications/preprocessing to the data as necessary. Finally, it returns the finished training and 
   testing data. 
   """
   df = shuffle(df)   
   x = df[inputs].iloc[:, :]
   y = df[outputs].iloc[:, :]
   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)
   
   t_list = list(set(outputs) - set(mod_attrs))
   t_list = [i for i in outputs if i not in mod_attrs]
   robust_scaler = RobustScaler().set_output(transform="pandas").fit(y_train[t_list])
   y_train = robust_scaler.transform(y_train[t_list])
   for v in mod_attrs:
      y_train.insert(outputs.index(v), v, df[v])
   y_train.columns = outputs
   joblib.dump(robust_scaler, "src/FUSION/fusionStandard.pkl")
   
   pd.DataFrame(x_test, columns=inputs).to_csv("src/FUSION/testData/x_test.csv")
   pd.DataFrame(y_test, columns=outputs).to_csv("src/FUSION/testData/y_test.csv")

   for idx, m in enumerate(func_attrs):
      modded = y_train[m].apply(funcs[idx])
      y_train[m] = modded

   return x_train, x_test, y_train, y_test


class UpdateHistory(callbacks.Callback):
    """
    This class defines the methods used to track validation losses across epochs so that the program 
    can reward the model for decreases in validation loss during training. 
    """
    def __init__(self):
        super(UpdateHistory, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        global prev_val_loss
        global curr_val_loss
        try:
            prev_val_loss.assign(curr_val_loss) 
            curr_val_loss.assign(logs["val_loss"])
        except KeyError:
            pass


@register_keras_serializable()
class LambdaLayerClass(Layer):
    """
    This class defines the operations that are used to inform the model of the physical laws that are 
    used to govern a stellar system so that it does not violate them when training and making 
    predictions. 
    """
    def __init__(self, func, name, **kwargs):
        super(LambdaLayerClass, self).__init__(**kwargs)
        self.name = name
        self.func = func

    def build(self, input_shape):
        super(LambdaLayerClass, self).build(input_shape)

    def call(self, inputs):
        if self.func == "mbol":
            return log10(inputs)
        elif self.func == "lbol":
            return log10(inputs[0]**2 * inputs[1]**4)
        elif self.func == "mass":
            return inputs ** (2/7)
        elif self.func == "density":
            return inputs[0] / inputs[1]
        elif self.func == "cpres":
            return log10(inputs[0]**2 / inputs[1]**4)
        elif self.func == "ctemp":
            return inputs[0] / inputs[1]
        elif self.func == "lifespan":
            return inputs[0] / inputs[1]
        elif self.func == "grav_bind":
            return log10(inputs[0]**2 / inputs[1])
        elif self.func == "fbol":
            return log10(inputs ** 4)
        elif self.func == "peak_wl":
            return 1 / inputs
        else:
            return inputs
    
    def compute_output_shape(self, input_shape):
        return TensorShape([None, 1])
    
    def get_config(self):
        config = super(LambdaLayerClass, self).get_config()
        config.update({
            "name": self.name, 
            "func": self.func
        })
        return config    
    
    @classmethod
    def from_config(cls, config):
        name = config["name"]
        func = config["func"]
        return cls(name=name, func=func)


def lambda_init(in_layer, indices, no_right=False):
   """
   This function formats the lambda layers (responsible for informing the model of the physical 
   formulas that govern stellar systems) to be compatible with the structure of the model defined in 
   the `createSubModel`, `createModels`, and `fuseModels` functions.
   """
   in_shape = in_layer.shape[1]
   inter1 = Reshape((in_shape, 1))(in_layer)
   
   raw_lambda_list = []
   for i in indices:
       if no_right:
           inter2 = Cropping1D(cropping=(i - 1, 0))(inter1)
       else:
           inter2 = Cropping1D(cropping=(i, in_shape - i - 1))(inter1)
       raw_lambda_list.append(inter2)
   reshaped_lambda_list = []
   for r in raw_lambda_list:
       inter3 = Reshape((1,))(r)
       reshaped_lambda_list.append(inter3)
   
   if len(reshaped_lambda_list) > 1:
       return reshaped_lambda_list
   return reshaped_lambda_list[0]


class ValLossRewardConstraint(Constraint):
    """
    The model is rewarded every time its validation loss decreases. This constraint sets limits on how 
    much the model can be rewarded based on how much the validation loss of the model changes 
    between training steps. This prevents the model’s loss from going out of control in the early 
    stages of training where changes in validation loss are more random, as the model has not looked 
    at enough data to see patterns in it. A side effect of this validation loss reward is that when the 
    validation loss increases, the model penalizes itself by adding to its loss.
    """    
    def __call__(self, weights):
        return clip_by_value(weights, 0.001, 3)


@register_keras_serializable()
class LossRewardOptimizer(Layer):
    '''
    This class defines the method used to train the variable that determines how much the model 
    should be rewarded based on how much validation loss improved between two steps.
    '''
    def __init__(self, name, **kwargs):
        super(LossRewardOptimizer, self).__init__(**kwargs)
        self.name = name

    def build(self, input_shape):
        def reward_initializer(shape, dtype=None):
            reward_init = np.mean(np.clip(np.random.normal(0.55, 0.2, 32), 0.001, 1))
            return constant(reward_init, shape=(1,), dtype=float32)
        
        self.lro_alpha = self.add_weight(name="lro_alpha", shape=(1,), initializer=reward_initializer, trainable=True, constraint=ValLossRewardConstraint())

        super(LossRewardOptimizer, self).build(input_shape)
    
    def call(self, inputs):
        return inputs

    def compute_output_shape(self, input_shape):
        return TensorShape(input_shape)

    def get_config(self):
        config = super(LossRewardOptimizer, self).get_config()
        config.update({
            "name": self.name,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(name=config["name"])


def RMSLE(y_true, y_pred):
   """
   This function defines the Square-Root Mean Squared Logarithmic Error loss function (among 
   other loss functions used such as Mean Squared Logarithmic Error and Categorical 
   Crossentropy, this one is not built in). It calculates the logarithmic difference between a predicted 
   value and the “true” value, squares that, finds the mean of each one of those values for all testing 
   data, and then takes the square root of the resultant value.
   """
   mean_squared_logarithmic_error = MSLE()
   return K.sqrt(mean_squared_logarithmic_error(y_true, y_pred))


@register_keras_serializable()
class DLR(Loss):
    """
    This class contains the methods used to implement the validation loss reward by subtracting it 
    from the models loss.
    """
    def __init__(self, init_loss_func, model, count, **kwargs):
        super(DLR, self).__init__(**kwargs)
        self.init_loss_func = init_loss_func
        self.model = model
        self.count = count

    def call(self, y_true, y_pred):
        global prev_val_loss
        global curr_val_loss
        val_loss_rewardRatio = self.model.get_layer("lroLayer{}".format(str(self.count))).lro_alpha
        init_loss = self.init_loss_func(y_true, y_pred)
        loss_reward = tanh(cast(val_loss_rewardRatio, float32)) * tanh((cast(prev_val_loss, float32) - cast(curr_val_loss, float32)) / cast(prev_val_loss, float32))
        final_loss = cast(init_loss, float32) - cast(loss_reward, float32)
        return convert_to_tensor(final_loss, dtype=float32)
    
    def get_config(self):
        config = super(DLR, self).get_config()
        config.update({
            "init_loss_func": self.init_loss_func,
            "model": None,
            "count": self.count
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        init_loss_func = config["init_loss_func"]
        model = config["model"]
        count = config["count"]
        return cls(init_loss_func=init_loss_func, model=model, count=count)


def lambda_functors():
   """
   This function initializes all of the specific physical formulas that govern stellar systems to be 
   added to the model by creating separate instances of the LamdaLayerClass class for each 
   formula involved.
   """
   mbol_lam = LambdaLayerClass(name="llcLayer0", func="mbol")
   lbol_lam = LambdaLayerClass(name="llcLayer1", func="lbol") 
   mass_lam = LambdaLayerClass(name="llcLayer2", func="mass")
   density_lam = LambdaLayerClass(name="llcLayer3", func="density")
   central_pressure_lam = LambdaLayerClass(name="llcLayer4", func="cpres")
   central_temp_lam = LambdaLayerClass(name="llcLayer5", func="ctemp")
   lifespan_lam = LambdaLayerClass(name="llcLayer6", func="lifespan")
   grav_bind_lam = LambdaLayerClass(name="llcLayer7", func="grav_bind")
   flux_lam = LambdaLayerClass(name="llcLayer8", func="fbol")
   peak_wavelength_lam = LambdaLayerClass(name="llcLayer9", func="peak_wl")

   return mbol_lam, lbol_lam, mass_lam, density_lam, central_pressure_lam, central_temp_lam, lifespan_lam, grav_bind_lam, flux_lam, peak_wavelength_lam


def createSubModel(shape=None, lambda_layer=None, lambda_inputs=None, norm=True, bound=None, embed=False, embed_dim=None, output_actv=None, output_neurons=1):
   """
   This function creates submodels for each parameter the final model predicts. Each submodel is 
   different as they have different tasks (some perform regression and others are classifiers), inputs, 
   formulas associated with them, and output types.
   """
   global gen_inputs
   global submodelCount

   if shape:
       input_layer = Input(shape=(shape,), name="input_layer{}".format(submodelCount))
       gen_inputs = input_layer
   else:
       input_layer = gen_inputs
   if embed:
       embedded = Embedding(input_dim=embed_dim, output_dim=2, embeddings_regularizer="L1L2", name="Embedding{}".format(submodelCount))(lambda_init(input_layer, embed, no_right=True))
       embedded = Flatten(name="Flatten{}".format(submodelCount))(embedded)
   else:
       embedded = input_layer
   if norm is True:
       norm_input = LayerNormalization(beta_regularizer="L1L2", gamma_regularizer="L1L2")(embedded)
   else:
       norm_input = Discretization(bin_boundaries=norm, output_mode="int", name="disc{}".format(submodelCount))(lambda_init(input_layer, bound))
       if embed:
          norm_input = concatenate([embedded, norm_input])
       norm_input = LayerNormalization(beta_regularizer="L1L2", gamma_regularizer="L1L2")(norm_input)

   hidden_input = Dense(32, activation=gelu, kernel_regularizer="L1L2", bias_regularizer="L1L2", activity_regularizer="L1L2", name="hidden_input{}".format(submodelCount))(norm_input)
   hidden_input = PReLU()(hidden_input)
   hidden_input = LayerNormalization(beta_regularizer="L1L2", gamma_regularizer="L1L2")(hidden_input)
   hidden_input = GaussianDropout(0.5)(hidden_input)

   if lambda_layer:
       selective_lambda_inputs_layer = lambda_init(embedded, lambda_inputs)
       selective_special_lambda_layer = lambda_layer(selective_lambda_inputs_layer)
       selective_hidden_special = Dense(32, activation=gelu, kernel_regularizer="L1L2", bias_regularizer="L1L2", activity_regularizer="L1L2", name="selective_hidden_special{}".format(submodelCount))(selective_special_lambda_layer)
       selective_hidden_special = PReLU()(selective_hidden_special)
       selective_hidden_special = LayerNormalization(beta_regularizer="L1L2", gamma_regularizer="L1L2")(selective_hidden_special)
       selective_hidden_special = GaussianDropout(0.5)(selective_hidden_special)

       combine = concatenate([hidden_input, selective_hidden_special], name="combine{}".format(submodelCount))
       combine = LayerNormalization(beta_regularizer="L1L2", gamma_regularizer="L1L2")(combine)
   else:
       combine = hidden_input

   hidden1 = Dense(64, activation=gelu, kernel_regularizer="L1L2", bias_regularizer="L1L2", activity_regularizer="L1L2", name="hidden1_{}".format(submodelCount))(combine)
   hidden1 = PReLU()(hidden1)
   hidden1 = LayerNormalization(beta_regularizer="L1L2", gamma_regularizer="L1L2")(hidden1)
   hidden1 = GaussianDropout(0.5)(hidden1)
   hidden2 = Dense(64, activation=gelu, kernel_regularizer="L1L2", bias_regularizer="L1L2", activity_regularizer="L1L2", name="hidden2_{}".format(submodelCount))(hidden1)
   hidden2 = PReLU()(hidden2)
   hidden2 = LayerNormalization(beta_regularizer="L1L2", gamma_regularizer="L1L2")(hidden2)
   hidden2 = GaussianDropout(0.5)(hidden2)
   hidden3 = Dense(64, activation=gelu, kernel_regularizer="L1L2", bias_regularizer="L1L2", activity_regularizer="L1L2", name="hidden3_{}".format(submodelCount))(hidden2)
   hidden3 = LayerNormalization(beta_regularizer="L1L2", gamma_regularizer="L1L2")(hidden3)
   hidden3 = PReLU()(hidden3)

   output = Dense(output_neurons, name="output_layer{}".format(submodelCount))(hidden3)
   if output_actv is not None:
      output = output_actv(output)
      output.name += str(submodelCount)

   lro_layer = LossRewardOptimizer(name="lroLayer{}".format(submodelCount))(output)
   gen_inputs = concatenate([gen_inputs, lro_layer], name="upd_inputs_with_new_output{}".format(submodelCount))
   submodelCount += 1

   return [input_layer, lro_layer]


def createModels():
   """
   This function calls the `createSubModel` function for each specific parameter the model 
   predicts and returns these separate submodels.
   """
   mbol_lam, lbol_lam, mass_lam, density_lam, central_pressure_lam, central_temp_lam, lifespan_lam, grav_bind_lam, flux_lam, peak_wavelength_lam = lambda_functors()

   mbol_submodel = createSubModel(shape=8, lambda_layer=mbol_lam, lambda_inputs=[1])
   absmag_submodel = createSubModel()
   lbol_submodel = createSubModel(lambda_layer=lbol_lam, lambda_inputs=[2, 0])
   mass_submodel = createSubModel(lambda_layer=mass_lam, lambda_inputs=[1])
   density_submodel = createSubModel(lambda_layer=density_lam, lambda_inputs=[11, 4])
   central_pressure_submodel = createSubModel(lambda_layer=central_pressure_lam, lambda_inputs=[11, 2])
   central_temp_submodel = createSubModel(lambda_layer=central_temp_lam, lambda_inputs=[11, 2])
   lifespan_submodel = createSubModel(lambda_layer=lifespan_lam, lambda_inputs=[11, 1])
   surf_grav_submodel = createSubModel()
   grav_bind_submodel = createSubModel(lambda_layer=grav_bind_lam, lambda_inputs=[11, 2])
   flux_submodel = createSubModel(lambda_layer=flux_lam, lambda_inputs=[0])
   metallicity_submodel = createSubModel()
   spectral_class_submodel = createSubModel(norm=[0., 4000., 5200., 7000., 12000., 20000., 34000., 420000.], bound=[0], embed=[20], embed_dim=7)
   lum_class_submodel = createSubModel(norm=[0., 25., 100., 1300., 8000., 125000.], bound=[1], embed=[21], embed_dim=7)
   peak_wavelength_submodel = createSubModel(lambda_layer=peak_wavelength_lam, lambda_inputs=[0])
   star_type_submodel = createSubModel(output_neurons=6, output_actv=Softmax())

   return [mbol_submodel, absmag_submodel, lbol_submodel, mass_submodel, density_submodel, central_pressure_submodel, central_temp_submodel, lifespan_submodel, surf_grav_submodel, grav_bind_submodel, flux_submodel, metallicity_submodel, spectral_class_submodel, lum_class_submodel, peak_wavelength_submodel, star_type_submodel]


def fuseModels(models, name):
   """
   This function combines (compiles) all of the submodels created from the `createModels`
   function into one model for training.
   """
   fusion_inputs = models[0][0]
   fusion_outputs = [y[1] for y in models]
   fusion = Model(inputs=fusion_inputs, outputs=fusion_outputs, name=name)
   loss_list = [DLR(MSLE(), fusion, 0), DLR(MSLE(), fusion, 1), DLR(RMSLE, fusion, 2), DLR(RMSLE, fusion, 3), DLR(MSLE(), fusion, 4), DLR(RMSLE, fusion, 5), DLR(RMSLE, fusion, 6), DLR(MSLE(), fusion, 7), DLR(RMSLE, fusion, 8), DLR(RMSLE, fusion, 9), DLR(RMSLE, fusion, 10), DLR(MAE(), fusion, 11), DLR(MSLE(), fusion, 12), DLR(MSLE(), fusion, 13), DLR(RMSLE, fusion, 14), DLR(CCE(from_logits=False, reduction="sum_over_batch_size"), fusion, 15)]
   fusion.compile(optimizer=Adam(learning_rate=0.0001), loss=loss_list, metrics=loss_list, run_eagerly=False, steps_per_execution=1, auto_scale_loss=True)
   return fusion


def Fuse():
   """
   This function is responsible for properly training the model using a multitude of methods (such 
   as EarlyStopping, which is responsible for stopping training when validation loss is no longer 
   improving and restoring the models best weights) and many fine-tuned hyperparameters.
   """
   dataset = pd.read_csv("src/FUSION/FusionStellaarData.csv")
   x_cols = ["EffectiveTemperature(Teff)(K)", "Luminosity(L/Lo)", "Radius(R/Ro)", "Diameter(D/Do)", "Volume(V/Vo)", "SurfaceArea(SA/SAo)", "GreatCircleCircumference(GCC/GCCo)", "GreatCircleArea(GCA/GCAo)"]
   y_cols = ["AbsoluteBolometricMagnitude(Mbol)", "AbsoluteMagnitude(M)(Mv)", "AbsoluteBolometricLuminosity(Lbol)(log(W))", "Mass(M/Mo)", "AverageDensity(D/Do)", "CentralPressure(log(N/m^2))", "CentralTemperature(log(K))", "Lifespan(SL/SLo)", "SurfaceGravity(log(g)...log(N/kg))", "GravitationalBindingEnergy(log(J))", "BolometricFlux(log(W/m^2))", "Metallicity(log(MH/MHo))", "SpectralClass", "LuminosityClass", "StarPeakWavelength(nm)", "StarType"]
   x_train, x_test, y_train, y_test = data_prep(dataset, x_cols, y_cols, ["SpectralClass", "LuminosityClass", "StarType"], ["StarType"], [lambda inpVec: to_categorical(inpVec, num_classes=6)])
   y_train = [np.stack(y_train[l]) for l in list(y_train)]

   Fusion = fuseModels(createModels(), name="Fusion")
   earlyStoppingCallback = callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=4, baseline=None, mode="min", verbose=2, restore_best_weights=True)
   tensorboard_callback = callbacks.TensorBoard(log_dir="src/FUSION/TensorBoardDataSummaries", update_freq=1000, write_images=True, write_steps_per_second=True, profile_batch=(11, 16))   
   Fusion.fit(x=x_train, y=y_train, validation_split=0.185, epochs=17, batch_size=128, shuffle=True, verbose=1, callbacks=[UpdateHistory(), callbacks.TerminateOnNaN(), earlyStoppingCallback, tensorboard_callback], validation_batch_size=32, validation_freq=1)
   Fusion.save("src/FUSION/fusionModel.keras")

   return Fusion, (x_test, y_test)


# To save output to a text file, run this file with '> src/FUSION/fusionTraining.txt' ('python src/FUSION/fusion.py > src/FUSION/fusionTraining.txt')
if __name__ == "__main__":
   Fuse()
