# Fusion
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
 
## The Project
The Fusion model is a deep neural network that can tackle regression and classification problems, depending on the parameter of a star it is predicting. Here are the required inputs and output parameters the model is compatible with: 
### Inputs (at least two of the following)
The target stars: 
- Effective Temperature (in Kelvin)
- Radius (in solar radii)
- Luminosity (in solar luminosity)
### Outputs
#### Outputs calculated outside of the model based on input values 
- Diameter (D/Do) (based on the radius input)
- Volume (V/Vo) (based on the radius input)
- Surface Area (SA/SAo) (based on the radius input)
- Great Circle Circumference (GCC/GCCo) (based on the radius input)
- Great Circle Area (GCA/GCAo) (based on the radius input)
#### Outputs predicted by the model (and whether they are a regression or classification prediction) 
- Absolute Bolometric Magnitude (Mbol) (regression)
- Absolute Magnitude (M)(Mv) (regression)
- Absolute Bolometric Luminosity (Lbol) (log(W)) (regression)
- Mass (M/Mo) (regression)
- Average Density (D/Do) (regression)
- Central Pressure (log(N/m^2)) (regression)
- Central Temperature (log(K)) (regression)
- Lifespan (SL/SLo) (regression)
- SurfaceGravity (log(g)...log(N/kg)) (regression)
- Gravitational Binding Energy (log(J)) (regression)
- Bolometric Flux (log(W/m^2)) (regression)
- Metallicity (log(MH/MHo)) (regression)
- Spectral Class (classification) (class M, K, G, F, A, B, or O) 
- Luminosity Class (classification) (class D, V, IV, III, II, Ib, or Ia)
- Star Peak Wavelength (nm) (regression)
- Star Type (classification) (class Brown Dwarf, Red Dwarf, White Dwarf, Main Sequence, Supergiant, or Hypergiant)
#### Outputs calculated outside the model but based on model predictions 
- Bolometric Correction (based on the predicted bolometric and absolute magnitude values) 
- Stellar Classification (based on the predicted Spectral and Luminosity classes. For example, the Sun is a G2V star—the '2' is a subclass (from 0–9, where 0 is the hottest sublass and 9 is the coolest) of the 'G' spectral class, and the 'V' is the star's luminosity class)

The model uses a TensorFlow/Keras backend with custom functions tailored to the specific regression and classification problems described above as well as overall model improvement. 

## Usage
### Install the application
#### Once the project materials are finished, I will start working on deployment and usage instructions for the general public 
### Clone the repository
#### Please take a look at the license found in this directory under the filename `LICENSE` 
### Contact Information: 
