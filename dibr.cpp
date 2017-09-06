#include <iostream>
#include <string>
#include <boost/python.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/legacy/legacy.hpp"
#include "conversion.h"

namespace py = boost::python;


py::tuple inverseMapping(PyObject* src_host_py) {
    NDArrayConverter cvt;
    try {
        cv::Mat src_host { cvt.toMat(src_host_py) };
        //cv::Mat src_host = cv::imread(std::string("img/0000.png"),CV_LOAD_IMAGE_GRAYSCALE);        
        cv::gpu::GpuMat dst, src;
        src.upload(src_host);
        cv::gpu::threshold(src, dst, 128.0, 255.0, CV_THRESH_BINARY);
        cv::Mat result_host;
        dst.download(result_host);
        return py::make_tuple(py::handle<>(cvt.toNDArray(result_host)));
    } catch(const cv::Exception& ex) {
        std::cout << "Error: " << ex.what() << std::endl;
    }
    return py::make_tuple(py::handle<>(cvt.toNDArray(cv::Mat())));
}

BOOST_PYTHON_MODULE(dibr)
{
    py::def("inverseMapping", inverseMapping);
}
