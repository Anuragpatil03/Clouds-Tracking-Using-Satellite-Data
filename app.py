import streamlit as st
from streamlit_option_menu import option_menu
import xarray as xr
import numpy as np
import pandas as pd
import os
import tobac
import iris
import matplotlib.pyplot as plt
import iris.plot as iplt
import iris.quickplot as qplt
import shutil
import cartopy.crs as ccrs
from pathlib import Path
# from six.moves import urllib
from netCDF4 import Dataset
import h5netcdf.legacyapi as netCDF4,h5netcdf
from IPython.display import HTML, Image, display
import base64
import tempfile
import matplotlib.animation as animation

st.title('IITM CLOUD TRACKING')
st.text(' Web App for Cloud Identification & Tracking with Radar Data ')


selected2 = option_menu(None, ["Home", "Info",], 
    icons=['house',"list-task"], 
    menu_icon="cast", default_index=0, orientation="horizontal")
selected2


st.sidebar.markdown("## Controls")
st.sidebar.markdown("You can **change** the values to change the *chart*.")
vmax = st.sidebar.number_input('vmax', min_value=0, max_value=100,value=17)
stub = st.sidebar.number_input('stubs', min_value=0, max_value=10,value=2)
order = st.sidebar.number_input('order', min_value=0, max_value=10,value=2)
exp = st.sidebar.number_input('extrapolate', min_value=0, max_value=10,value=0)
memory = st.sidebar.number_input('memory', min_value=0, max_value=10,value=1)
adstop = st.sidebar.number_input('adaptive stop', min_value=0.01, max_value=1.0,value=0.2)
adstep = st.sidebar.number_input(' adaptive step', min_value=0.01, max_value=2.00,value=0.95)
subsize = st.sidebar.number_input('subnetwork size', min_value=1, max_value=1000,value=100)

import warnings
warnings.filterwarnings('ignore', category=UserWarning, append=True)
warnings.filterwarnings('ignore', category=RuntimeWarning, append=True)
warnings.filterwarnings('ignore', category=FutureWarning, append=True)
warnings.filterwarnings('ignore',category=pd.io.pytables.PerformanceWarning)


def trigger(OLR,vmax,stub,order,exp,memory,adstop,adstep,subsize ):


    
    savedir=Path("Save")
    if not savedir.is_dir():
        savedir.mkdir()
    plot_dir=Path("Plot")
    if not plot_dir.is_dir():
        plot_dir.mkdir()

    dxy,dt=tobac.get_spacings(OLR,grid_spacing=1000)

    parameters_features={}
    parameters_features['position_threshold']='weighted_diff'   
    parameters_features['sigma_threshold']=0.5
    parameters_features['n_min_threshold']=4
    parameters_features['target']='minimum'
    parameters_features['threshold']=[50,45,40,35,30]
    print('starting feature detection')
    st.info('Starting feature detection')
    Features=tobac.feature_detection_multithreshold(OLR,dxy, **parameters_features)
    Features.to_hdf(savedir / 'Features.h5', 'table')
    print('feature detection performed and saved')
    st.success('Feature detection performed and saved')

    parameters_segmentation={}
    parameters_segmentation['target']='minimum'
    parameters_segmentation['method']='watershed'
    parameters_segmentation['threshold']=50

    print('Starting segmentation based on OLR.')
    st.info('Starting segmentation')
    Mask_OLR,Features_OLR=tobac.segmentation_2D(Features,OLR,dxy,**parameters_segmentation)
    print('segmentation OLR performed, start saving results to files')
    iris.save([Mask_OLR], savedir / 'Mask_Segmentation_OLR.nc', zlib=True, complevel=4)
    Features_OLR.to_hdf(savedir / 'Features_OLR.h5', 'table')
    print('segmentation OLR performed and saved')
    st.success('Segmentation performed and saved')

    parameters_linking={}
    parameters_linking['v_max']=vmax #17 #20                                 
    parameters_linking['stubs']=stub  #2                         
    parameters_linking['order']=order  #2 #1
    parameters_linking['extrapolate']=exp  #0 #2
    parameters_linking['memory']= memory      #1  #0 
    parameters_linking['adaptive_stop']= adstop   #0.2
    parameters_linking['adaptive_step']=adstep    #0.95
    parameters_linking['subnetwork_size']=subsize   #100
    parameters_linking['method_linking']= 'predict'

    Track=tobac.linking_trackpy(Features, OLR, dt=dt, dxy=dxy, **parameters_linking)
    Track.to_hdf(savedir / 'Track.h5', 'table')

    axis_extent=[74.9,76.8,16.8,18.6]

    fig_map,ax_map=plt.subplots(figsize=(10,10),subplot_kw={'projection': ccrs.PlateCarree()})

    ax_map=tobac.map_tracks(Track,axis_extent=axis_extent,axes=ax_map)
    st.info("Features")

    
    animation_test_tobac=tobac.animation_mask_field(Track,Features,OLR,Mask_OLR, axis_extent=axis_extent,vmin=80,vmax=330,
    cmap='Blues_r',plot_outline=True,plot_marker=True,marker_track='x',plot_number=True,plot_features=True)

    # writervideo = animation.FFMpegWriter(fps=5)
    # animation_test_tobac.save('increasingStraightLine.mp4', writer=writervideo)
    
    st.dataframe(Features)
    st.info("Displaying Video...")
    savefile_animation=os.path.join(plot_dir,'Animation.mp4')
    animation_test_tobac.save(savefile_animation,dpi=200)
    print('animation saved to {savefile_animation}')
    # v=HTML(animation_test_tobac.to_html5_video())
    st.video("Plot/Animation.mp4")
    st.success("Cloud Tracking Performed")
    
if selected2 == "Info":
        st.title("Parameters: ")
        st.info("frame: Frame/time/file number; starts from 0 and increments by 1 to N times.")
        st.info("idx: Feature number within that frame; starts at 1, increments by 1 to the number of features for each frame, and resets to 1 when the frame increments")
        st.info("hdim_1: First horizontal dimension in grid point space (typically, although not always, N/S or y space).")
        st.info("hdim_2: Second horizontal dimension in grid point space (typically, although not always, E/W or x space)")
        st.info("num: Number of grid points that are within the threshold of this feature")
        st.info("threshold_value: Maximum threshold value reached by the feature")
        st.info("feature: Unique number of the feature; starts from 1 and increments by 1 to the number of features identified in all frames")
        st.info("time: Time of the feature")
        st.info("timestr: String representation of the feature time : YYYY-MM-DD HH:MM:SS")
        st.info("y: Grid point y location of the feature (see hdim_1 and hdim_2)")
        st.info("x:	Grid point x location of the feature (see also y)")
        st.info("projection_y_coordinate: Y location of the feature in projection coordinates")
        st.info("projection_x_coordinate: X location of the feature in projection coodinates")
        st.info("lat: Latitude of the feature")
        st.info("lon: Longitude of the feature")

if selected2 == "Home":
    upload_file = st.file_uploader('**Upload a NC File**')
    if upload_file is not None:
        file_details = {"FileName":upload_file.name,"FileType":upload_file.type}
        st.write(file_details)
        with open(os.path.join("tempDir",upload_file.name),"wb") as f: 
            f.write(upload_file.getbuffer())         
        st.success("Saved File")
        filename = "tempDir/"+upload_file.name
        cube = iris.load_cube(filename,'zmax')
        print(cube)
        trigger(cube,vmax,stub,order,exp,memory,adstop,adstep,subsize)


    # f = netCDF4.Dataset(upload_file, mode='r')
    # # print(f.variables)
   

    