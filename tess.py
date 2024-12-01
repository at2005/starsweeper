from astroquery.mast import Observations
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time
import numpy as np
from astroquery.mast import Tesscut
from astropy.io import fits
import matplotlib.pyplot as plt

def fetch_tess_sector5_data(center_x=1450, center_y=1024, size=100):
    """
    Fetch TESS image data from Sector 5, Camera 1, CCD 4 centered around specified coordinates.
    
    Parameters:
    -----------
    center_x : int
        X coordinate in CCD pixels (default: 1450)
    center_y : int
        Y coordinate in CCD pixels (default: 1024)
    size : int
        Size of the cutout region in pixels (default: 100)
    
    Returns:
    --------
    tuple
        (image_data, time_array, wcs) if successful, (None, None, None) if no data found
    """
    try:
        # Define the time range for Sector 5
        # Using the time from the table: 2458461.19 JD
        target_time = Time(2458461.19, format='jd')
        time_range = [target_time.jd - 11, target_time.jd + 11]  # roughly ±11 days
        
        print(f"Searching for data in Sector 5 around pixel coordinates: ({center_x}, {center_y})")
        print(f"Time range: {time_range}")
        
        # Query TESS observations
        obs_query = Observations.query_criteria(
            obs_collection='TESS',
            dataproduct_type='image',
            t_min=time_range[0],
            t_max=time_range[1],
            sequence_number=5  # Sector 5
        )
        
        if len(obs_query) == 0:
            print("No observations found for specified criteria")
            return None, None, None
            
        print(f"Found {len(obs_query)} matching observations")
        
        # Get cutout using pixel coordinates
        try:
            # Modified to specify camera and CCD
            cutouts = Tesscut.get_cutouts(
                coordinates=SkyCoord(ra=57*u.deg, dec=7.6*u.deg),  # approximate center
                size=size,
                sector=5,
                cameras=[1],  # Camera 1
                ccds=[4]      # CCD 4
            )
            
            if not cutouts:
                print("No cutout data returned")
                return None, None, None
                
            print("Successfully retrieved cutout data")
            
            hdulist = cutouts[0]
            data_table = hdulist[1].data
            
            # Get time array
            time_array = data_table['TIME'] + 2457000  # Convert to JD
            
            # Get WCS information
            wcs = None
            if 'WCSAXES' in hdulist[2].header:
                from astropy.wcs import WCS
                wcs = WCS(hdulist[2].header)
            
            # Get all flux data
            flux_data = data_table['FLUX']
            
            return flux_data, time_array, wcs
            
        except Exception as e:
            print(f"Error during cutout processing: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None, None, None
            
    except Exception as e:
        print(f"Error in fetch_tess_sector5_data: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None, None, None

def plot_tess_time_series(flux_data, time_array, save_path=None):
    """
    Plot a time series of TESS images.
    
    Parameters:
    -----------
    flux_data : numpy.ndarray
        Array of flux data for multiple timestamps
    time_array : numpy.ndarray
        Array of observation times in JD
    save_path : str, optional
        Path to save the animation (if None, display instead)
    """
    if flux_data is None:
        print("No flux data to plot")
        return
        
    # Create a figure to show images at different times
    n_times = min(5, len(time_array))  # Show up to 5 timestamps
    indices = np.linspace(0, len(time_array)-1, n_times, dtype=int)
    
    fig, axes = plt.subplots(1, n_times, figsize=(20, 4))
    
    for i, idx in enumerate(indices):
        im = axes[i].imshow(flux_data[idx], origin='lower', cmap='gray')
        axes[i].set_title(f'Time: {time_array[idx]:.2f} JD')
        plt.colorbar(im, ax=axes[i], label='Flux')
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    print("Starting TESS data fetch for Sector 5...")
    
    # Fetch data around the cutout origin from the table
    flux_data, time_array, wcs = fetch_tess_sector5_data(
        center_x=1450,
        center_y=1024,
        size=100  # Adjust size as needed
    )
    
    if flux_data is not None:
        print("Data successfully retrieved, creating plots...")
        # Plot time series
        plot_tess_time_series(flux_data, time_array)
    else:
        print("Failed to retrieve TESS data")