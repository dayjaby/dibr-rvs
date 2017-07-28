import numpy as np
import OpenEXR

def exportGrayEXR(filename,data, size):
    tempData = np.array(data,dtype=np.float32).reshape(-1).tostring()
    out = OpenEXR.OutputFile(filename,OpenEXR.Header(*size))
    out.writePixels({'R': tempData, 'G': tempData, 'B': tempData})


