import numpy as np
import zarr
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm import tqdm

if (__name__ == '__main__'):
    cat = np.loadtxt('sky_ldev_truthcat.txt', skiprows=1)[:, 1:]
    n_sources = cat.shape[0]

    f = fits.open('sky_ldev.fits')

    wcs = WCS(f[0].header)

    nf, ny, nx = f[0].data.shape

    fout = zarr.open('training.zarr', 'w')

    ds_images = fout.create_dataset('images', shape=(n_sources, 32*32*128), dtype=np.float32)

    batchsize = 64

    n_batches = n_sources // batchsize
    n_remaining = n_sources % batchsize    

    images = np.zeros((batchsize, 32*32*128), dtype=np.float32)

    loop = 0

    # All batches

    for batch in tqdm(range(n_batches), desc='batch'):

        lowb = loop
        highb = loop + batchsize

        for i in tqdm(range(batchsize), desc='object', leave=False):

            ra = cat[loop, 0] * u.deg
            dec = cat[loop, 1] * u.deg
            coords = SkyCoord(ra, dec, unit="deg")
            freq = cat[loop, 4] * u.Hz
            coords = wcs.world_to_pixel(coords, freq)

            
            lowx = int(coords[0]) - 16
            highx = int(coords[0]) + 16        
            if (lowx < 0):
                delta = -lowx
                lowx += delta
                highx += delta
            if (highx >= nx):
                delta = highx - nx
                lowx -= delta
                highx -= delta

            lowy = int(coords[1]) - 16
            highy = int(coords[1]) + 16
            if (lowy < 0):
                delta = -lowy
                lowy += delta
                highy += delta
            if (highy >= ny):
                delta = highy - ny
                lowy -= delta
                highy -= delta
                
            lowf = int(coords[2]) - 64
            highf = int(coords[2]) + 64        
            if (lowy < 0):
                delta = -lowf
                lowf += delta
                highf += delta
            if (highf >= nf):
                delta = highf - nf
                lowf -= delta
                highf -= delta
            
            cube = f[0].data[lowf:lowf+128, lowy:highy, lowx:highx].reshape((128*32*32))

            images[i, :] = cube

            loop += 1
        
        ds_images[lowb:highb, :] = images[:]

    # Remaining images

    lowb = loop
    highb = loop + n_remaining

    images = np.zeros((n_remaining, 32*32*128), dtype=np.float32)

    for i in tqdm(range(n_remaining), desc='object', leave=False):

        ra = cat[loop, 0] * u.deg
        dec = cat[loop, 1] * u.deg
        coords = SkyCoord(ra, dec, unit="deg")
        freq = cat[loop, 4] * u.Hz
        coords = wcs.world_to_pixel(coords, freq)

        
        lowx = int(coords[0]) - 16
        highx = int(coords[0]) + 16        
        if (lowx < 0):
            delta = -lowx
            lowx += delta
            highx += delta
        if (highx >= nx):
            delta = highx - nx
            lowx -= delta
            highx -= delta

        lowy = int(coords[1]) - 16
        highy = int(coords[1]) + 16
        if (lowy < 0):
            delta = -lowy
            lowy += delta
            highy += delta
        if (highy >= ny):
            delta = highy - ny
            lowy -= delta
            highy -= delta
            
        lowf = int(coords[2]) - 64
        highf = int(coords[2]) + 64        
        if (lowy < 0):
            delta = -lowf
            lowf += delta
            highf += delta
        if (highf >= nf):
            delta = highf - nf
            lowf -= delta
            highf -= delta
        
        cube = f[0].data[lowf:lowf+128, lowy:highy, lowx:highx].reshape((128*32*32))

        images[i, :] = cube

        loop += 1

        ds_images[lowb:highb, :] = images[:]
