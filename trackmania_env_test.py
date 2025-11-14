# Trackmania number: 2225070

from tmrl import get_environment
from time import sleep
import numpy as np

# actions are [gas, break, steer], gas = {0,1}, break = {0,1}, steer between -1.0 and +1.0
def random_model(obs):
    # Creates completely random data to see if trackmania works with random inputs
    steer = np.random.uniform(-1,1)
    return np.array([1.0, 0.0, steer])

env = get_environment()

sleep(2.0)  # just so we have time to focus the TM20 window after starting the script

obs, info = env.reset() 
for _ in range(200): 
    act = random_model(obs) 
    # print(act)
    obs, rew, terminated, truncated, info = env.step(act) 
    # Observations (Given TMFULL)
    # Obs[0] = (1,) speed
    # Obs[1] = (1,) gear
    # Obs[2] = (1,) RPM
    # Obs[3] = (4,64,64) Grayscale Image
    # Obs[4] = (3,) ?
    # Obs[5] = (3,) ?

    # look to see if the shapes match above
    print("Type of obs:", type(obs))
    if isinstance(obs, tuple):
        print("Tuple length:", len(obs))
        for i, o in enumerate(obs):
            if hasattr(o, "shape"):
                print(f"obs[{i}] shape:", o.shape, "dtype:", o.dtype)
            else:
                print(f"obs[{i}] type:", type(o), "value:", o)
    if terminated or truncated:
        break
env.unwrapped.wait()



