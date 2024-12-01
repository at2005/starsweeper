import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
from astroquery.jplhorizons import Horizons
from datetime import datetime, timedelta

# Sedna's JPL Small-Body ID is 90377
survey_range = [2010 + i for i in range(5)]

# Lists to store results
ra_list = []
dec_list = []


def get_sedna_icrs(date_str):
    # t = Time(f"{year}-01-01 00:00:00")
    t = Time(date_str)
    # Query JPL Horizons for Sedna's position
    obj = Horizons(id='90377',  # Sedna's JPL ID
                   location='500@0',  # Geocentric observer
                   epochs=t.jd)  # Julian Date
    
    # Get ephemerides
    eph = obj.ephemerides()
    
    # Extract RA and Dec (in degrees)
    ra = float(eph['RA'][0])
    dec = float(eph['DEC'][0])
    return ra,dec

for year in survey_range:
    date_str = f"{year}-01-01 00:00:00"
    ra, dec = get_sedna_icrs(date_str)
    ra_list.append(ra)
    dec_list.append(dec)
    
    # print(f"Year {year}:")
    # print(f"RA: {ra:.4f}°")
    # print(f"Dec: {dec:.4f}°\n")


def julian_date_to_datetime(jd):
    # Use the correct time scale for Julian Dates
    t_tt = Time(jd, format='mjd', scale='tt')
    # Convert from TT to UTC
    t_utc = t_tt.utc
    # Get the ISO formatted time
    iso_time = t_utc.isot
    return iso_time

# print(get_sedna_icrs(julian_date_to_datetime(55935.23363)))
# print(get_sedna_icrs(julian_date_to_datetime(55935.24578)))
# print(get_sedna_icrs(julian_date_to_datetime(55935.35509)))
# print(get_sedna_icrs(julian_date_to_datetime(56991.32707)))

from astroquery.mast import Observations
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time
import numpy as np
from astroquery.mast import Tesscut
from astropy.io import fits
import matplotlib.pyplot as plt

