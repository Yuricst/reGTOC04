# generally full of helper functions

import pandas as pd

from orbit import Orbit, AU2km, mu, day2s

def load_dataset(file = '../gtoc4_problem_data.txt'):

    df = pd.read_csv(file, skiprows=2,delimiter= '\s+',header=None)
    df.columns = ['Name','Epoch','a','e','i','LAN','omega','M0']

    asteroids = set()

    for i in range(len(df)):
        o = Orbit(index=i)
        o.from_gtoc(*df.iloc[i])
        asteroids.add(o)

    earth = create_earth()

    return asteroids, earth



def create_earth():
    # create Earth
    earth = Orbit(name='Earth', index=-1)
    earth.from_gtoc('Earth',54000,0.999988049532578, 1.671681163160e-2, 0.8854353079654e-3, 175.40647696473, 287.61577546182, 257.60683707535)

    return earth
