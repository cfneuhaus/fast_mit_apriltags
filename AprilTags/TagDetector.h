#ifndef TAGDETECTOR_H
#define TAGDETECTOR_H

#include <vector>

#include "opencv2/opencv.hpp"

#include "AprilTags/FloatImage.h"
#include "AprilTags/TagDetection.h"
#include "AprilTags/TagFamily.h"

#include "AprilTags/Edge.h"
#include "AprilTags/ThreadPool.h"

namespace AprilTags
{

class TagDetector
{
public:
    const TagFamily thisTagFamily;

    //! Constructor
    // note: TagFamily is instantiated here from TagCodes
    TagDetector(const TagCodes& tagCodes)
        : thisTagFamily(tagCodes)
    {
        // these are just rough guesses
        const int expected_width = 1024;
        const int expected_height = 1024;

        const int numThreads = (int)threadPool.getNumThreads();
        edgeArrs.resize(numThreads);
        int packageSize = std::max(1, (expected_height - 1) / int(numThreads));
        for (int i = 0; i < (int)threadPool.getNumThreads(); i++)
            edgeArrs[i].reserve(expected_width * packageSize * 4);

        edges.reserve(expected_width * expected_height * 4);
        storage.resize(expected_width * expected_height * 4);
    }

    std::vector<TagDetection> extractTags(const cv::Mat& image);
    int verifyQuad(const std::vector<std::pair<float, float>>& p, const cv::Mat& gray);

private:
    ThreadPool threadPool;


    std::vector<Edge> edges;
    std::vector<std::vector<Edge>> edgeArrs;

    std::vector<float> storage;
};

} // namespace

#endif
