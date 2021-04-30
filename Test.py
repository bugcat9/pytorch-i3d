import numpy as np

file1 = "THUMOS14/flow/video_test_0000004.npy"
file2 = "flow_feature/video_test_0000004.npy"
file3 = "THUMOS14/flow/video_test_0000006.npy"
file4 = "flow_feature/video_test_0000006.npy"

a = np.load(file1)
b = np.load(file2)
c = np.load(file3)
d = np.load(file4)
print()
