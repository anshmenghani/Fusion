# Calculates star attributes based on input data and adds them to the input data csv file 

import pandas as pd
import numpy as np
import scipy.constants as c

path = "src/FUSION/total_combined_data.csv"
df = pd.read_csv(path)

df = df.drop("Unnamed: 0", axis=1)
df.columns = ["EffectiveTemperature(Teff)(K)", "AbsoluteMagnitude(M)(Mv)", "SurfaceGravity(log(g)...log(N/kg))", "Radius(R/Ro)", "Metallicity(log(MH/MHo))"]
new_order = ["EffectiveTemperature(Teff)(K)", "Radius(R/Ro)", "AbsoluteMagnitude(M)(Mv)", "SurfaceGravity(log(g)...log(N/kg))", "Metallicity(log(MH/MHo))"]
df = df[new_order]

R_sun = 6.96e8
L_sun = 3.827e26
M_sun = 1.989e30


def get_lum(teff, r):
    t_list = teff.to_list()
    r_list = r.to_list()
    l_list = [(4 * np.pi * (r_list[i] * R_sun)**2 * c.sigma * (t_list[i]**4)) / L_sun for i in range(len(t_list))]
    return l_list

def mbol(lum):
    l_list = lum.to_list()
    m_list = [4.8 + ((-2.5*np.log10(lum[i]))-0.05) for i in range(len(l_list))]
    return m_list

def bc(mv, mbolometric):
    mv_list = mv.to_list()
    mbol_list = mbolometric.to_list()
    bc_list = [mv_list[i] - mbol_list[i] for i in range(len(mv_list))]
    return bc_list

def lbol(sa, teff):
    sa_list = sa.to_list()
    teff_list = teff.to_list()
    lbol_list = [np.log10((sa_list[i]*6090000000000) * (c.sigma*teff_list[i]**4)) for i in range(len(sa_list))]
    return lbol_list

def mass(lum):
    l_list = lum.to_list()
    mass_list = [l_list[i]**(2/7) for i in range(len(l_list))]
    return mass_list

def ad(mass, vol):
    mass_list = mass.to_list()
    vol_list = vol.to_list()
    ad_list = [mass_list[i] / vol_list[i] for i in range(len(mass_list))]
    return ad_list

def cpres(mass, rad):
    mass_list = mass.to_list()
    rad_list = rad.to_list()
    cpres_list = [np.log10((3*c.G*(M_sun*mass_list[i])**2) / (8*np.pi*(R_sun*rad_list[i])**4)) for i in range(len(mass_list))]
    return cpres_list

def ctemp(mass, rad):
    mass_list = mass.to_list()
    rad_list = rad.to_list()
    ctemp_list = [np.log10((c.G*c.u*(M_sun*mass_list[i])) / (c.k*R_sun*rad_list[i])) for i in range(len(mass_list))]
    return ctemp_list

def lifespan(mass, lum):
    mass_list = mass.to_list()
    lum_list = lum.to_list()
    lifespan_list = [mass_list[i] / lum_list[i] for i in range(len(mass_list))]
    return lifespan_list

def grav_bind(mass, rad):
    mass_list = mass.to_list()
    rad_list = rad.to_list()
    grav_list = [np.log10((3*c.G*(mass_list[i]*M_sun)**2) / (5*R_sun*rad_list[i])) for i in range(len(mass_list))]
    return grav_list

def fbol(teff):
    teff_list = teff.to_list()
    fbol_list = [np.log10(c.sigma * teff_list[i]**4) for i in range(len(teff_list))]
    return fbol_list

def star_type(teff, lum):
    teff_list = teff.to_list()
    lum_list = lum.to_list()
    def get_type(t, l):
        if t < 2400:
            return 0
        elif t >= 2400 and t < 4500 and l <= 25:
            return 1
        elif l < 0.1:
            return 2
        elif t > 4500 and t < 30000 and l <= 25:
            return 3
        elif l > 25 and l <= 125000:
            return 4
        elif l > 125000:
            return 5
    st_list = [get_type(teff_list[i], lum_list[i]) for i in range(len(teff_list))]
    return st_list

def lum_class(lum):
    lum_list = lum.to_list()
    def get_lc(l):
        if l <= 0.1:
            return ["D", 0]
        elif l <= 25:
            return ["V", 6]
        elif l <= 100:
            return ["IV", 5]
        elif l <= 1300:
            return ["III", 4]
        elif l <= 8000:
            return ["II", 3]
        elif l <= 125000:
            return ["Ib", 2]
        elif l > 125000:
            return ["Ia", 1]
    lc_list = [get_lc(lum_list[i]) for i in range(len(lum_list))]
    return lc_list

def spec_class(teff):
    teff_list = teff.to_list()
    def get_sc(t):
        if t < 4000:
            return ["M" + str(int(np.floor((t - 2400) / 160))), 0]
        elif t < 5200:
            return ["K" + str(int(np.floor((t - 4000) / 120))), 1]
        elif t < 7000:
            return ["G" + str(int(np.floor((t - 5200) / 200))), 2]
        elif t < 12000:
            return ["F" + str(int(np.floor((t - 7000) / 500))), 3]
        elif t < 20000:
            return ["A" + str(int(np.floor((t - 12000) / 800))), 4]
        elif t < 34000:
            return ["B" + str(int(np.floor((t - 20000) / 1400))), 5]
        elif t <= 420000:
            return ["O" + str(int(np.floor((t - 34000) / 76000))), 6]
    sc_list = [get_sc(teff_list[i]) for i in range(len(teff_list))]
    return sc_list

def s_classif(sc, lc):
    sc_list = sc.to_list()
    lc_list = lc.to_list()
    scif_list = [str(sc_list[i]) + str(lc_list[i]) for i in range(len(sc_list))]
    return scif_list


df.insert(1, "Luminosity(L/Lo)", get_lum(df["EffectiveTemperature(Teff)(K)"], df["Radius(R/Ro)"]))
df.insert(3, "Diameter(D/Do)", [r for r in df["Radius(R/Ro)"].to_list()])
df.insert(4, "Volume(V/Vo)", [r**3 for r in df["Radius(R/Ro)"].to_list()])
df.insert(5, "SurfaceArea(SA/SAo)", [r**2 for r in df["Radius(R/Ro)"].to_list()])
df.insert(6, "GreatCircleCircumference(GCC/GCCo)", [r for r in df["Radius(R/Ro)"].to_list()])
df.insert(7, "GreatCircleArea(GCA/GCAo)", [r**2 for r in df["Radius(R/Ro)"].to_list()])
df.insert(8, "AbsoluteBolometricMagnitude(Mbol)", mbol(df["Luminosity(L/Lo)"]))
df.insert(10, "BolometricCorrection(BC)(mag)", bc(df["AbsoluteMagnitude(M)(Mv)"], df["AbsoluteBolometricMagnitude(Mbol)"]))
df.insert(11, "AbsoluteBolometricLuminosity(Lbol)(log(W))", lbol(df["SurfaceArea(SA/SAo)"], df["EffectiveTemperature(Teff)(K)"]))
df.insert(12, "Mass(M/Mo)", mass(df["Luminosity(L/Lo)"]))
df.insert(13, "AverageDensity(D/Do)", ad(df["Mass(M/Mo)"], df["Volume(V/Vo)"]))
df.insert(14, "CentralPressure(log(N/m^2))", cpres(df["Mass(M/Mo)"], df["Radius(R/Ro)"]))
df.insert(15, "CentralTemperature(log(K))", ctemp(df["Mass(M/Mo)"], df["Radius(R/Ro)"]))
df.insert(16, "Lifespan(SL/SLo)", lifespan(df["Mass(M/Mo)"], df["Luminosity(L/Lo)"]))
df.insert(18, "GravitationalBindingEnergy(log(J))", grav_bind(df["Mass(M/Mo)"], df["Radius(R/Ro)"]))
df.insert(19, "BolometricFlux(log(W/m^2))", fbol(df["EffectiveTemperature(Teff)(K)"]))
df.insert(21, "SpectralClass", [v[1] for v in spec_class(df["EffectiveTemperature(Teff)(K)"])])
df.insert(22, "LuminosityClass", [v[1] for v in lum_class(df["Luminosity(L/Lo)"])])
df.insert(23, "StarPeakWavelength(nm)", [2897771.955/i for i in df["EffectiveTemperature(Teff)(K)"].to_list()])
df.insert(24, "StarType", star_type(df["EffectiveTemperature(Teff)(K)"], df["Luminosity(L/Lo)"]))
df.insert(25, "LuminosityClass_aux", [v[0] for v in lum_class(df["Luminosity(L/Lo)"])])
df.insert(26, "SpectralClass_aux", [v[0] for v in spec_class(df["EffectiveTemperature(Teff)(K)"])])
df.insert(25, "StellarClassification", s_classif(df["SpectralClass_aux"], df["LuminosityClass_aux"]))
df.to_csv("FusionStellaarData.csv", index=False)
