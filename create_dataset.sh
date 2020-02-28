#!/usr/bin/env bash

scenflow_data_path=path-to-dataset/SceneFlow

monkaa_frames_cleanpass=$scenflow_data_path"/monkaa/frames_cleanpass"
monkaa_disparity=$scenflow_data_path"/monkaa/disparity"
driving_frames_cleanpass=$scenflow_data_path"/driving/frames_cleanpass"
driving_disparity=$scenflow_data_path"/driving/disparity"
flyingthings3d_frames_cleanpass=$scenflow_data_path"/flyingthings3d/frames_cleanpass"
flyingthings3d_disparity=$scenflow_data_path"/flyingthings3d/disparity"

mkdir dataset

ln -s $monkaa_frames_cleanpass dataset/monkaa_frames_cleanpass
ln -s $monkaa_disparity dataset/monkaa_disparity
ln -s $flyingthings3d_frames_cleanpass dataset/frames_cleanpass
ln -s $flyingthings3d_disparity dataset/frames_disparity
ln -s $driving_frames_cleanpass dataset/driving_frames_cleanpass
ln -s $driving_disparity dataset/driving_disparity

