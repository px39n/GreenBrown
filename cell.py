from greenbrown.bfast.breakpoints import extract_bps
from greenbrown.utils import load_example
import numpy as np
if __name__ == '__main__':
    ndvi_s=load_example()
    y=ndvi_s.values
    x=np.arange(1, len(y) + 1)
    a=extract_bps(y,5)
    print(a)
