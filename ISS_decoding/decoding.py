import sys
import os
from typing import Mapping, Tuple, Union
import numpy as np
from starfish.types import Axes, Coordinates, Features, Number
from starfish import Codebook
from starfish.experiment.builder import FetchedTile, TileFetcher
from slicedimage import ImageFormat
from skimage.io import imread
from starfish.experiment.builder import write_experiment_json
#import matplotlib.pyplot as plt
import starfish
from copy import deepcopy
from starfish import data, FieldOfView, display, Experiment
from starfish.image import Filter
from starfish.spots import DetectPixels
from starfish.types import Axes, Features, Levels
from starfish import data, FieldOfView
from starfish.image import ApplyTransform, Filter, LearnTransform, Segment
from starfish.spots import FindSpots, DecodeSpots, AssignTargets
from starfish.types import Axes, FunctionSource, Levels
from starfish.core.expression_matrix.expression_matrix import ExpressionMatrix
from starfish.core.intensity_table.intensity_table import IntensityTable
test = os.getenv("TESTING") is not None
from starfish.core.spots.DecodeSpots.trace_builders import build_spot_traces_exact_match
import pandas as pd
from starfish.types import Axes, TraceBuildingStrategies
import warnings
warnings.filterwarnings('ignore')
from starfish.image import ApplyTransform, Filter, LearnTransform, Segment
from starfish.spots import FindSpots, DecodeSpots, AssignTargets
from starfish.types import Axes, FunctionSource
#import matplotlib
#import matplotlib.pyplot as plt
import pprint
from starfish.types import Features, Axes
#from starfish.util.plot import imshow_plane
test = os.getenv("TESTING") is not None

def ISS_pipeline(fov, codebook,
                register = True, 
                masking_radius = 15, 
                threshold = 0.002, 
                sigma_vals = [1, 10, 30], # min, max and number
                decode_mode = 'PRMC' # or MD
                ):

    print('getting images')
    primary_image = fov.get_image(FieldOfView.PRIMARY_IMAGES) # primary images
    print('creating reference images')
    dots = primary_image.reduce({Axes.CH, Axes.ROUND}, func="max") # reference round for image registration
    # register the raw image
    if register == True: 
        print('registering images')
        learn_translation = LearnTransform.Translation(reference_stack=dots, axes=Axes.ROUND, upsampling=100)
        transforms_list = learn_translation.run(primary_image.reduce({Axes.CH, Axes.ZPLANE}, func="max"))
        warp = ApplyTransform.Warp()
        registered = warp.run(primary_image, transforms_list=transforms_list,  in_place=False, verbose=True)
        # filter registered data
        masking_radius = masking_radius
        filt = Filter.WhiteTophat(masking_radius, is_volume=False)
        #filtered = filt.run(primary_image, verbose=True, in_place=False)
        filtered = filt.run(registered, verbose=True, in_place=False)
    else: 
        print('not registering images')
        # filter raw data
        masking_radius = masking_radius
        filt = Filter.WhiteTophat(masking_radius, is_volume=False)
        filtered = filt.run(primary_image, verbose=True, in_place=False)
        
    # normalize the channel intensities
    print('normalizing channel intensities')
    sbp = starfish.image.Filter.ClipPercentileToZero(p_min=80, p_max=99.999, level_method=Levels.SCALE_BY_CHUNK)
    scaled = sbp.run(filtered, n_processes = 1, in_place=False)
    
    bd = FindSpots.BlobDetector(
        min_sigma=sigma_vals[0],
        max_sigma=sigma_vals[1],
        num_sigma=sigma_vals[2],
        threshold=threshold, # this is set quite low which means that we will capture a lot of signals
        measurement_type='mean',
    )
    
    # detect spots using laplacian of gaussians approach
    dots_max = dots.reduce((Axes.ROUND, Axes.ZPLANE), func="max")
    print('locating spots')
    # locate spots in a reference image
    spots = bd.run(reference_image=dots_max, image_stack=scaled)
    
    if decode_mode == 'PRMC':
        print('decoding with PerRoundMaxChannel')
        decoder = DecodeSpots.PerRoundMaxChannel(codebook=codebook)
        decoded = decoder.run(spots=spots)
        # Build IntensityTable with same TraceBuilder as was used in MetricDistance

            
    elif decode_mode == 'MD':
        print('decoding with MetricDistance')
    # decode the pixel traces using the codebook
        decoder = DecodeSpots.MetricDistance(
            codebook=experiment.codebook,
            max_distance=1,
            min_intensity=1,
            metric='euclidean',
            norm_order=2,
            trace_building_strategy=TraceBuildingStrategies.EXACT_MATCH
        )
        decoded = decoder.run(spots=spots)
    
    intensities = build_spot_traces_exact_match(spots)

    # Get vector magnitudes, deal with empty tiles
    if intensities.size == 0: 
        print('No spots found')
    else:
        norm_intensities, vector_magnitude = codebook._normalize_features(intensities, norm_order=2)
    
    # Get distances
    distances = decoded.to_decoded_dataframe().data['distance'].to_numpy()

    return decoded

def process_experiment(exp_path, 
                        output, 
                        register = True, 
                        masking_radius = 15, 
                        threshold = 0.002, 
                        sigma_vals = [1, 10, 30], # min, max and number
                        decode_mode = 'PRMC' # or MD
                ):
    
    # create output folder if not exists
    if not os.path.exists(output):
        os.makedirs(output)
    
    # load experiment file
    experiment = Experiment.from_json(exp_path)
    all_fovs = list(experiment.keys())

    # from output, find the FOVs not processed 
    csv_files = sorted(os.listdir(output))
    try:
        fovs_done = list(pd.DataFrame(csv_files)[0].str.split('.',expand = True)[0])
    except KeyError:
        print('no FOVS done')
        fovs_done = []
    
    # specify the files not done
    not_done = sorted(set(all_fovs).difference(set(fovs_done)))
    
    for i in not_done:
        print('decoding '+i)
        decoded = ISS_pipeline(experiment[i], experiment.codebook, register, masking_radius, threshold, sigma_vals, decode_mode)
        df = decoded.to_features_dataframe()
        df.to_csv(output + i +'.csv')

