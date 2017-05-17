 #!/bin/bash

 # ---------------------------------------------------------------------------
 # Video Propagation Networks
 #----------------------------------------------------------------------------
 # Copyright 2017 Max Planck Society
 # Distributed under the BSD-3 Software license [see LICENSE.txt for details]
 # ---------------------------------------------------------------------------

set -e
set -x

FOLDID=$1
TRAIN_LOG_DIR="../data/training_data/training_logs/"

for STAGEID in 1
do
  LOG_FILE=$TRAIN_LOG_DIR'FOLD'$FOLDID'_STAGE'$STAGEID'.log'
  echo $FOLDID
  echo $STAGEID
  echo $LOG_FILE

  if [ $# -eq 2 ]
  then
    python train_online_seg.py $FOLDID $STAGEID $2
  else
    python train_online_seg.py $FOLDID $STAGEID
  fi
  
  # python select_model.py $FOLDID $STAGEID || exit 1 &&
  # python run_segmentation.py $STAGEID $FOLDID || exit 1
done

set +x