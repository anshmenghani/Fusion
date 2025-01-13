# Fusion
The Fusion Stellaar Model is designed to predict and classify
stellar parameters with higher accuracy than traditional formulas. 
It takes inputs of a star's Effective Temperature, Luminosity, and 
Radius, and runs its Deep Neural Network algorithm to predict and 
classify other parameters of a star such as Mass, Surface Gravity, 
etc. It is trained on the ESA's Gaia mission dataset with over 403 
million features.  

## The Project
The Fusion model is a deep neural network that can tackle regression and classification problems, depending on the parameter of a star it is predicting. Here are the required inputs and output parameters the model is compatible with: 
### Inputs (at least two of the following)
The target star's: 
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
- Spectral Class (classification) 
- Luminosity Class (classification)
- Star Peak Wavelength (nm) (regression)
- Star Type (classification)
#### Outputs calculated outside the model but based on model predictions 
- Bolometric Correction (based on the predicted bolometric and absolute magnitude values) 
- Stellar Classification (based on the predicted Spectral and Luminosity classes)

The model uses a TensorFlow/Keras backend with custom functions tailored to the specific regression and classification problems described above as well as overall model improvement. The model is tested to have a [_] percent average accuracy per round of predictions and a [_] average speed per round of predictions.

## Usage
#### Once paper is finished I will start working on deployment and usage instructions for the general public 
