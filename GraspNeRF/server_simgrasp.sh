#!/bin/bash

'''
This was the run script with the settings 
for running the first set of experiments
for the first integration configuration
'''

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

GPUID=1
BLENDER_BIN="$ROOT_DIR/blender/blender"
BLENDER_PY="$ROOT_DIR/blender/2.93/python/bin/python3.9"

# Use Conda env
VENV_PATH="/home/jeziel_a/miniconda3/envs/graspVenv"
VENV_SITEPKGS="$VENV_PATH/lib/python3.9/site-packages"

# Redirect Blender's Python to use venv site-packages
export PYTHONPATH="$VENV_SITEPKGS:$PYTHONPATH"

# Project-specific
RENDERER_ASSET_DIR="$SCRIPT_DIR/data/assets"
BLENDER_PROJ_PATH="$SCRIPT_DIR/data/assets/material_lib_graspnet-v2.blend"
SIM_LOG_DIR="$SCRIPT_DIR/log/$(date '+%Y%m%d-%H%M%S')"


scene="single" 
object_set="vrb_experiments"
material_type="transparent" # this change from: diffuse, transparent, specular
render_frame_list="2,6,10,14,18,22"
check_seen_scene=0
expname=single_trans_vrb

NUM_TRIALS=50
METHOD='graspnerf'

mycount=0 
while (( $mycount < $NUM_TRIALS )); do

    $BLENDER_BIN $BLENDER_PROJ_PATH --background --python-expr "import site; site.addsitedir('$VENV_SITEPKGS')" \
    --python scripts/sim_grasp.py -- \
    $mycount $GPUID $expname $scene $object_set $check_seen_scene $material_type \
    $RENDERER_ASSET_DIR $SIM_LOG_DIR 0 $render_frame_list $METHOD

    python ./scripts/stat_expresult.py -- $SIM_LOG_DIR $expname
    ((mycount=$mycount+1))
done
