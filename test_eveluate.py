"""
Script to test pretrained models.
"""
from pathlib import Path
import os
from tensorflow.keras.optimizers import SGD
import numpy as np

import matplotlib.pyplot as plt

from plenpy.spectral import SpectralImage

import lfcnn
from lfcnn.metrics import get_disparity_metrics, get_central_metrics_small
from lfcnn.models import get_model_type

# Set up path to model weights
model_weights_path = Path("weights/mt_uncertainty_al_normgradsim_regular.h5")

# Set up paths to data
data_test_path = Path("/mnt/symphony/wen/lightfield_DataSet/PATCHED_F_9x9_36x36_one/test.h5")
data_evaluate_path = Path("/mnt/symphony/wen/lightfield_DataSet/CHALLENGES/9x9_CHALLENGE_MS_LF_DISP_CENTRAL.h5")
# Set up dataset properties
# Adapt if using a different dataset than the one provided
range_data = 2**16 - 1  # 16 Bit data
data_key = "lf"
label_key = "disp"
augmented_shape = (9, 9, 32, 32, 13)
generated_shape = (32, 32, 9*9, 13)  # Adapt if using different angular resolution
augmented_shape_full = (9, 9, 512, 512, 13)
generated_shape_full = (512, 512, 9*9, 13)
# Number of GPUs to use
gpus = 1

# Set up mini-batch size
batch_size =1 #64

# Whether to code incoming data and with which mask
# Adapt if testing other coding masks
# See here for available masks:
# https://gitlab.com/iiit-public/lfcnn/-/blob/master/lfcnn/generators/utils.py#L383
use_mask = True
mask_type = "random"

verbose = True

# Set up data preprocessing multiprocessing
use_multiprocessing = True
workers = 4
max_queue_size = 4

# Setup test kwargs to pass to model.test()
test_kwargs = dict(data=data_test_path,
                   data_key=data_key,
                   label_keys=label_key,
                   augmented_shape=augmented_shape,
                   generated_shape=generated_shape,
                   range_data=range_data,
                   batch_size=batch_size,
                   gpus=gpus,
                   use_mask=use_mask,
                   mask_type=mask_type,
                   fix_seed=True,
                   workers=workers,
                   max_queue_size=max_queue_size,
                   use_multiprocessing=use_multiprocessing,
                   verbose=verbose
                   )

# Setup test kwargs to pass to model.evaluate_challenges()
eval_kwargs = dict(data=data_evaluate_path,
                       data_key=data_key,
                       label_keys=label_key,
                       augmented_shape=augmented_shape_full,
                       generated_shape=generated_shape_full,
                       range_data=range_data,
                       use_mask=use_mask,
                       mask_type=mask_type,
                       fix_seed=True,
                       workers=workers,
                       max_queue_size=max_queue_size,
                       use_multiprocessing=use_multiprocessing,
                       verbose=verbose
                       )


# Set up dummy optimizer for model
optimizer = SGD(1e-3)

# Set up metrics for validation and testing
metrics = dict(disparity=get_disparity_metrics(),
               central_view=get_central_metrics_small())

# Initialize and comile model
model = "conv3ddecode2d"  # Adapt if using 4D conv model
model_type = "center_and_disparity"
model_kwargs = dict(num_filters_base=24, skip=True, kernel_reg=1e-5)

model = get_model_type(model_type).get(model)
model = model(**model_kwargs,
              loss=None, callbacks=None,
              optimizer=optimizer, metrics=metrics)

model.load_weights(str(model_weights_path),
                   generated_shape=generated_shape,
                   augmented_shape=augmented_shape,
                   by_name=True,
                   skip_mismatch=True)

tempdir_path = Path('/mnt/symphony/wen/lfcnn/reconstruction-from-spectrally-coded-light-fields-main/output')
# Test
print("Testing...")
# test_res = model.test(**test_kwargs)
print("... done.")

print("Evaluating challenges...")
eval_res = model.evaluate_challenges(**eval_kwargs)
print("... done")

# Test results
print("Showing test results...")
# for key in test_res:
 #    print("test_" + key, test_res[key])
print("... done.")

  # Iterate over all scenes
for i, metrics in enumerate(eval_res['metrics']):
     # Iterate over metrics
    for key in metrics:
        print("eval"+key, metrics[key], i)
    print("... done.")

print("Saving results...")
#tempdir_path = Path(os.environ['/mnt/symphony/wen/lfcnn/reconstruction-from-spectrally-coded-light-fields-main/output'])
predict_path = tempdir_path / f'predict.npz'
np.savez(predict_path, **eval_res)

 # Add PNGs as artifacts for challenges
if 'central_view' in eval_res.keys():
    for i in range(len(eval_res['central_view'])):
        # Convert central view reconstruction to RGB and save as PNG
        central_path = tempdir_path/f"_central_rgb.png"
        SpectralImage(eval_res['central_view'][i]).save_rgb(central_path)
        # Add artifacts
        #_run.add_artifact(central_path)

if 'disparity' in eval_res.keys():
    for i in range(len(eval_res['disparity'])):
        # Create PNG for disparity reconstruction
        disp_path = tempdir_path/f"_disparity_rgb.png"

        fig = plt.figure(frameon=True)
        fig.set_size_inches(1, 1)
        ax = plt.Axes(fig, [0.08, 0.16, 0.82, 0.82])
        ax.set_axis_off()
        fig.add_axes(ax)
        im = ax.imshow(np.squeeze(eval_res['disparity'][i]), interpolation='none', aspect='auto')
        cb = fig.colorbar(im, orientation='horizontal', cax=fig.add_axes([0.3, 0.1, 0.4, 0.03]))
        cb.ax.tick_params(labelsize=2, size=1, width=0.1, pad=0.2)
        cb.outline.set_linewidth(0.01)
        fig.savefig(disp_path, dpi=512)

        # Add artifacts
        #_run.add_artifact(disp_path)


print("... done.")


