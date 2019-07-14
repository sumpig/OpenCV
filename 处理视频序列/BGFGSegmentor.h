#if !defined BGFGSeg
#define BGFGSeg

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "videoprocessor.h"

class BGFGSegmentor : public FrameProcessor {

	cv::Mat gray;           // current gray-level image
	cv::Mat background;     // accumulated background
	cv::Mat backImage;      // current background image
	cv::Mat foreground;     // foreground image
	double learningRate;    // learning rate in background accumulation
	int threshold;          // threshold for foreground extraction


public:

	BGFGSegmentor() : threshold(10), learningRate(0.01) {}

	void setThreshold(int t) {

		threshold = t;
	}

	void setLearningRate(double r) {

		learningRate = r;
	}

	void process(cv::Mat& frame, cv::Mat& output) {

		cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

		// initialize backround to 1st frame
		if (background.empty())
			gray.convertTo(background, CV_32F);

		background.convertTo(backImage, CV_8U);

		cv::absdiff(backImage, gray, foreground);
		cv::threshold(foreground, output, threshold, 255, cv::THRESH_BINARY_INV);

		// accumulate background
		cv::accumulateWeighted(gray, background, learningRate, output);
	}
};

#endif
