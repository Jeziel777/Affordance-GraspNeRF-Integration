#!/bin/bash

'''
This was the run script with the settings 
for running the baseline experiments

'''

# Resolve the directory where the script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"  # This points to GraspNeRF's parent: GraspNerf2

# Activate the correct virtualenv site-packages
export PYTHONPATH="$ROOT_DIR/graspVenv/lib/python3.9/site-packages:$PYTHONPATH"

GPUID=0
BLENDER_BIN="$ROOT_DIR/blender/blender"

RENDERER_ASSET_DIR="$SCRIPT_DIR/data/assets"
BLENDER_PROJ_PATH="$SCRIPT_DIR/data/assets/material_lib_graspnet-v2.blend"
SIM_LOG_DIR="$SCRIPT_DIR/log/$(date '+%Y%m%d-%H%M%S')"

scene="single"
object_set="vrb_experiments"
material_type="diffuse" # this change from: diffuse, transparent, specular
render_frame_list="2,6,10,14,18,22"
check_seen_scene=0
expname=single_vrb_diffuse_single_view

NUM_TRIALS=50
METHOD='graspnerf'


mycount=0
while (( $mycount < $NUM_TRIALS )); do
   $BLENDER_BIN $BLENDER_PROJ_PATH --background --python scripts/sim_grasp.py \
   -- $mycount $GPUID $expname $scene $object_set $check_seen_scene $material_type \
   $RENDERER_ASSET_DIR $SIM_LOG_DIR 0 $render_frame_list $METHOD

   python ./scripts/stat_expresult.py -- $SIM_LOG_DIR $expname
((mycount=$mycount+1));
done;
