Queen Mary Multi-Camera Distributed Traffic Scenes Dataset (QMDTS) is released for research purpose only.
http://www.eecs.qmul.ac.uk/~xx302
by Alex Xun Xu
16, Sep 2015


List of contents:

# QMDTS_Videos.zip ---- contains train and test raw videos for 27 scenes introduced in [1]. Train videos are used to learn local topics while test videos are used for behaviour profiling, query by example, cross-scene classification and multi-scene summarization.

# ReferenceAnnotation.mat ---- contains the annotations for 112 clips for each of Scene 7, 8, 9, 10, 18 and 19.

Feature Extraction:

It is recommended to use Liu Ce's optical flow code [2] for computing visual features. A python implementation can be found at [3]. The codebook construction and bag of words feature generation can be found in the paper [1] Experiment section.

Ref:
[1] X. Xu, T. Hospedales, S. Gong, Discovery of Shared Semantic Spaces for Multi-Scene Video Query and Summarization, IEEE Transactions om Circuits and Systems for Video Technology
[2] http://people.csail.mit.edu/celiu/OpticalFlow/
[3] https://pythonhosted.org/bob.ip.optflow.liu/py_api.html