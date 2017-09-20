#!/bin/bash

#floyd run --env caffe:py2 "python2 app.py"
floyd run --env caffe:py2 --data suchecki/datasets/photo-critic-weights/1:weights --mode serve

