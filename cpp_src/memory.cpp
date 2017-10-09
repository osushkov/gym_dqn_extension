
#include <iostream>
#include <string>
#include <sstream>
#include <vector>

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

namespace np = boost::python::numpy;
namespace bp = boost::python;

struct World {
     std::string mMsg;

    void set(std::string msg) { mMsg = msg; }

    void many(boost::python::list msgs) {
        long l = len(msgs);
        std::stringstream ss;
        for (long i = 0; i<l; ++i) {
            if (i>0) ss << ", ";
            std::string s = boost::python::extract<std::string>(msgs[i]);
            ss << s;
        }
        mMsg = ss.str();
    }

    std::string greet() { return mMsg; }
};

BOOST_PYTHON_MODULE(hello_ext)
{
    //np::initialize();
    
    bp::class_<World>("World")
        .def("greet", &World::greet)
        .def("set", &World::set)
        .def("many", &World::many)
    ;
}
