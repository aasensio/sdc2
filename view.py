import numpy as np
import napari
from astropy.io import fits

f = fits.open('sky_ldev.fits')

with napari.gui_qt():
    viewer = napari.view_image(np.log(np.abs(f[0].data)))