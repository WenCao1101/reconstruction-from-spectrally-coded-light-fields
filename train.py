"""
Train script
"""
from pathlib import Path

from tensorflow_addons.optimizers import Yogi

import lfcnn
from lfcnn.callbacks import SigmoidDecay
from lfcnn.losses import get as get_loss
from lfcnn.metrics import get_disparity_metrics, get_central_metrics_small
from lfcnn.models import get_model_type
from lfcnn.training import get_model_cls

# Set up paths to data and data shapes
base_path = Path("/mnt/symphony/wen/lightfield_DataSet/PATCHED_F_9x9_36x36_test")
data_train_path = base_path / "train.h5"
data_valid_path = base_path / "validation.h5"
data_test_path = base_path / "test.h5"

# Set up dataset properties
# Adapt if using a different dataset than the one provided
range_data = 2**16 - 1  # 16 Bit data
data_key = "lf"
label_key = "disp"
augmented_shape = (9, 9, 32, 32, 13)
generated_shape = (32, 32, 9*9, 13)  # Adapt if using different angular resolution

# Shuffle data (training dataset only)
shuffle = True

# Number of GPUs to use
gpus = 1

# Set up training hyperparameters
batch_size = 2
epochs = 170
learning_rate = 5e-3

# Data augmentation
augment = False

# Whether to code incoming data and with which mask
# Adapt if testing other coding masks
# See here for available masks:
# https://gitlab.com/iiit-public/lfcnn/-/blob/master/lfcnn/generators/utils.py#L383
use_mask = True
mask_type = "random"

verbose = 1

# Set up data preprocessing multiprocessing
use_multiprocessing = True
workers = 4
max_queue_size = 4

# Setup train kwargs to pass to model.train()
train_kwargs = dict(data=data_train_path,
                    valid_data=data_valid_path,
                    data_key=data_key,
                    label_keys=label_key,
                    augmented_shape=augmented_shape,
                    generated_shape=generated_shape,
                    range_data=range_data,
                    range_valid_data=range_data,
                    batch_size=batch_size,
                    valid_batch_size=batch_size,
                    epochs=epochs,
                    shuffle=shuffle,
                    augment=augment,
                    gpus=gpus,
                    use_mask=use_mask,
                    mask_type=mask_type,
                    workers=workers,
                    max_queue_size=max_queue_size,
                    use_multiprocessing=use_multiprocessing,
                    verbose=verbose
                    )

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

# Set up loss for training with multitask uncertainty (MT Uncertainty)
# as well as Normalized Gradient Similarity (AL NormGradSim)
loss_disp = get_loss("Huber")(delta=1.0, ver='lfcnn')
loss_central = get_loss("Huber")(delta=1.0, ver='lfcnn')
loss = dict(disparity=loss_disp, central_view=loss_central)

model_cls = get_model_cls("NormalizedGradientSimilarity")
model_cls_kwargs = dict(aux_losses=dict(disparity=["WeightedTotalVariation", "DisparityNormalSimilarity"],
                                        central_view=["NormalizedStructuralSimilarity", "NormalizedCosineProximity"]),
                        gradient_approximation="None",
                        multi_task_uncertainty=True)

# Set up metrics for validation and testing
metrics = dict(disparity=get_disparity_metrics(),
               central_view=get_central_metrics_small())

# Set up optimizer and learning rate decay
optimizer = Yogi(learning_rate)
lr_sched = SigmoidDecay(lr_init=learning_rate,
                        max_epoch=epochs-1,
                        alpha=0.1,
                        offset=int(0.6*epochs),
                        lr_min=2e-2*learning_rate)
callbacks = [lr_sched]

# Initialize and comile model
model = "conv3ddecode2d"
model_type = "center_and_disparity"
model_kwargs = dict(num_filters_base=24, skip=True, kernel_reg=1e-5)

model = get_model_type(model_type).get(model)
model = model(**model_kwargs,
              optimizer=optimizer, loss=loss, metrics=metrics, callbacks=callbacks,
              model_cls=model_cls, model_cls_kwargs=model_cls_kwargs)

# Train and validate
print("Training...")
hist = model.train(**train_kwargs)
print("... done.")

# Test
print("Testing...")
test_res = model.test(**test_kwargs)
print("... done.")

# Test results
print("Showing test results...")
for key in test_res:
    print("test_" + key, test_res[key])
print("... done.")

# Optional: save model weights
# print("Saving model weights...")
# model.save_weights("weights.h5")
# print("... done.")
