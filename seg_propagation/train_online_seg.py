#!/usr/bin/env python

'''
    File name: train_online_seg.py
    Author: Varun Jampani
'''

# ---------------------------------------------------------------------------
# Video Propagation Networks
#----------------------------------------------------------------------------
# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license [see LICENSE.txt for details]
# ---------------------------------------------------------------------------

import numpy as np
import sys
import os

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib', 'caffe', 'build', 'tools'))

from init_caffe import *
from create_solver import *
from create_online_net import *
from davis_data import *

num_frames = NUM_PREV_FRAMES

def Train_segmentation(fold_id, stage_id, resume=True):

    lr = float(0.001)
    # prefix = INTER_MODEL_DIR + 'FOLD' + fold_id + '_' + 'STAGE' + stage_id
    display=10
    test_iter = 50
    iter_size = 4
    test_interval = 200
    num_iter = 15000
    snapshot_iter = 200
    debug_info = False
    train_net_file = get_bnn_cnn_train_net_fold_stage(num_frames, fold_id, stage_id, phase = 'TRAIN')
    test_net_file = get_bnn_cnn_train_net_fold_stage(num_frames, fold_id, stage_id, phase = 'TEST')
    gpus = "1"


    import shutil
    job_dir = os.path.join('..', 'jobs', 'fold_%s_stage_%s' % (fold_id, stage_id) )
    if not os.path.exists(job_dir):
      os.makedirs(job_dir)

    new_train_net_file = os.path.join(job_dir, 'train.prototxt')
    new_test_net_file = os.path.join(job_dir, 'test.prototxt')
    
    shutil.copy(train_net_file, new_train_net_file)
    shutil.copy(test_net_file, new_test_net_file)
    
    prefix = os.path.join(os.path.abspath(job_dir), 'snapshots', 'FOLD' + fold_id + '_' + 'STAGE' + stage_id)

    solver_proto = create_solver_proto(new_train_net_file,
                                       new_test_net_file,
                                       lr,
                                       prefix,
                                       display=display,
                                       test_iter = test_iter,
                                       test_interval = test_interval,
                                       max_iter=num_iter,
                                       iter_size=iter_size,
                                       snapshot=snapshot_iter,
                                       debug_info=debug_info)
    
    solver_file = os.path.join(job_dir, 'solver.prototxt')
    # solver = create_solver(solver_proto)
    print('Writing', solver_file)
    with open(solver_file, 'w') as fp:
        fp.write(str(solver_proto))

    
    if int(stage_id) > 1:
        init_model = SELECTED_MODEL_DIR + 'FOLD' + fold_id + '_' + 'STAGE' +\
            str(int(stage_id) - 1) + '.caffemodel'
    else:
        init_model = SELECTED_MODEL_DIR + 'deeplab_vpn_init_model.caffemodel'

    # solver.net.copy_from(init_model)
    # solver.solve()

    ### To use multi-gpu,
    prefix = 'FOLD' + fold_id + '_' + 'STAGE' + stage_id
    train(job_dir, solver_file, num_iter, prefix, init_model, gpus, resume)


def train(job_dir, solver_file, max_iter, prefix, init_model, gpus, resume):
        
    job_file = "{}/train.sh".format(job_dir)
    
    max_iter = 0
    snapshot_dir = "{}/snapshots".format(job_dir)
    if not os.path.exists(snapshot_dir):
      os.makedirs(snapshot_dir)

    if resume:        
        # Find most recent snapshot.
        for file in os.listdir(snapshot_dir):
          if file.endswith(".solverstate"):
            basename = os.path.splitext(file)[0]
            iter = int(basename.split("{}_iter_".format(prefix))[1])
            if iter > max_iter:
              max_iter = iter

    train_src_param = '--weights="{}" \\\n'.format(init_model)
    if resume:
        if max_iter > 0:
            train_src_param = '--snapshot="{}/{}_iter_{}.solverstate" \\\n'.format(snapshot_dir, prefix, max_iter)
   
    # Create job file.
    caffe_bin = os.path.join('..', 'lib', 'caffe', 'build', 'tools', 'caffe')
    with open(job_file, 'w') as f:
        f.write('export PYTHONPATH="%s/seg_propagation:${PYTHONPATH}"\n'%os.path.abspath('..'))
        f.write('{} train \\\n'.format(caffe_bin))
        f.write('--solver="{}" \\\n'.format(solver_file))
        f.write(train_src_param)
        if resume:
            f.write('--gpu {} 2>&1 | tee -a {}/train_{}.log\n'.format(gpus, job_dir, prefix))
        else:
            f.write('--gpu {} 2>&1 | tee {}/train_{}.log\n'.format(gpus, job_dir, prefix))
            
            # parsed_log_file = glob.glob("{}*.train".format(job_dir))
            # if len(parsed_log_file) > 0 and os.path.exists(parsed_log_file[0]):
            #     old_dir = "{}old_log".format(job_dir)
            #     if not os.path.exists(old_dir):
            #         os.makedirs(old_dir)
            #     shutil.copy(parsed_log_file[0], old_dir)
            #     os.remove(parsed_log_file[0])
            # parsed_log_file = glob.glob("{}*.test".format(job_dir))
            # if len(parsed_log_file) > 0 and os.path.exists(parsed_log_file[0]):
            #     old_dir = "{}old_log".format(job_dir)
            #     if not os.path.exists(old_dir):
            #         os.makedirs(old_dir)
            #     shutil.copy(parsed_log_file[0], old_dir)
            #     os.remove(parsed_log_file[0])
    
    import stat
    import subprocess

    os.chmod(job_file, stat.S_IRWXU)
    subprocess.call(job_file, shell=True)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: ' + sys.argv[0] + ' <fold_id> <stage_id>')
    elif len(sys.argv) == 4:
        Train_segmentation(str(sys.argv[1]), str(sys.argv[2]), int(sys.argv[3]))
    else:
        Train_segmentation(str(sys.argv[1]), str(sys.argv[2]))
