Queen Mary Multi-Camera Distributed Traffic Scenes Dataset (QMDTS)[1] is released for research purpose only.
http://xu-xun.com
by Xun Xu
2 May 2019

####### Dataset

The original videos for QMDTS dataset can be downloaded from https://www.dropbox.com/s/rcr2amj8mu46jm3/QMDTS_Videos.rar?dl=0 .
This file contains training and testing raw videos for 27 scenes introduced in [1]. Train videos are used to learn local topics while test videos are used for behaviour profiling, query by example, cross-scene classification and multi-scene summarization.

The annotations for 112 clips for each of the scene 7, 8, 9, 10, 18 and 19 can be found at https://www.dropbox.com/s/zqza5tjfwh80ali/ReferenceAnnotation.mat?dl=0

List of all contents:

# QMDTS_Videos.zip ---- training & testing raw videos for 27 scenes

# ReferenceAnnotation.mat ----the annotations for 112 clips for each of Scene 7, 8, 9, 10, 18 and 19.

####### Details of videos

Each scene consists of two separate videos named as Scn_x_Train.mp4 and Scn_x_Test.mp4 where x is the scene index. Each video has 9000 frames and should be further segmented into clips in the way introduced in section VII of [1].

It is recommend to extract the mp4 video into image sequence using ffmpeg as below
ffmpeg -i videoname.mp4 ./frame%06d.png



Reference
[1] X. Xu, T. Hospedales, S. Gong, Discovery of Shared Semantic Spaces for Multi-Scene Video Query and Summarization, IEEE Transactions om Circuits and Systems for Video Technology, 2017
