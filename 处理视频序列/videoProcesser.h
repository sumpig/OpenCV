#if !defined VPROCESSOR
#define VPROCESSOR

#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

// The frame processor interface
class FrameProcessor {

public:
	// processing method
	virtual void process(cv::Mat& input, cv::Mat& output) = 0;
};

class VideoProcessor {

private:

	// the OpenCV video capture object
	cv::VideoCapture capture;

	// the callback function to be called for the processing of each frame
	void (*process)(cv::Mat&, cv::Mat&);

	// the pointer to the class implementing the FrameProcessor interface
	FrameProcessor* frameProcessor;

	// a bool to determine if the process callback will be called
	bool callIt;

	// Input and Output display window name
	std::string windowNameInput;
	std::string windowNameOutput;

	// delay between each frame processing
	int delay;

	// number of processed frames
	long fnumber;

	// stop at this frame number
	long frameToStop;

	// to stop the processing
	bool stop;

	// vector of image filename to be used as input
	std::vector<std::string> images;

	// image vector iterator
	std::vector<std::string>::const_iterator itImg;

	// the OpenCV video writer object
	cv::VideoWriter writer;

	// output filename
	std::string outputFile;

	// current index for output images
	int currentIndex;

	// number of digits in output image filename
	int digits;

	// extension of output images
	std::string extension;

	// to get the next frame, could be video file; camera; vector of images
	bool readNextFrame(cv::Mat& frame) {

		if (images.size() == 0)
			return capture.read(frame);

		else {

			if (itImg != images.end()) {
				frame = cv::imread(*itImg);
				itImg++;
				return frame.data != 0;
			}

			return false;
		}
	}

	// to write the output frame, could be video file or images
	void writeNextFrame(cv::Mat& frame) {

		if (extension.length()) { // then we write images

			std::stringstream ss; // need #include <sstream>
			ss << outputFile << std::setfill('0') << std::setw(digits) << currentIndex++ << extension;
			cv::imwrite(ss.str(), frame);
		}
		else {

			writer.write(frame);
		}
	}

public:

	// Constructor setting the default values
	VideoProcessor() :
		callIt(false),
		delay(-1),
		fnumber(0),
		stop(false),
		digits(0),
		frameToStop(-1),
		process(0),
		frameProcessor(0) {}

	// set the name of the video file
	bool setInput(std::string filename) {

		fnumber = 0;

		// In case a resource was already associated with the VideoCapture instance
		capture.release();
		images.clear();

		// Open the video file
		return capture.open(filename);
	}

	// set the camera ID
	bool setInput(int id) {

		fnumber = 0;
		capture.release();
		images.clear();

		return capture.open(id);
	}

	// set the vector of input images
	bool setInput(const std::vector<std::string>& imgs) {

		fnumber = 0;
		capture.release();

		// the input will be this vector of images
		images = imgs;
		itImg = images.begin();

		return true;
	}

	// set the output video file by default the same parameters than input video will be used
	bool setOutput(const std::string& filename, int codec = 0, double framerate = 0.0, bool isColor = true) {

		outputFile = filename;
		extension.clear();

		if (framerate == 0.0)
			framerate = getFrameRate(); // same as input

		char c[4];
		if (codec == 0)
			codec = getCodec(c); // same as input

		// Open output video
		return writer.open(outputFile,
			codec,
			framerate,
			getFrameSize(), // frame size
			isColor);
	}

	// set the output as a series of image files,extension must be ".jpg",".bmp"...
	bool setOutput(const std::string& filename, // filename prefix
		const std::string& ext, // image file extension
		int numberOfDigits = 3, // number of digits
		int startIndex = 0) {  // start index

// number of digits must be positive
		if (numberOfDigits < 0)
			return false;

		// filenames and their common extension
		outputFile = filename;
		extension = ext;

		// number of digits in the file numbering scheme
		digits = numberOfDigits;

		// start numbering at this index
		currentIndex = startIndex;

		return true;
	}

	// set the callback function that will be called for each frame
	void setFrameProcessor(void (*frameProcessingCallback) (cv::Mat&, cv::Mat&)) {

		// invalidate frame processor class instance
		frameProcessor = 0;

		// this is the frame processor function that will be called
		process = frameProcessingCallback;
		callProcess();
	}

	// set the instance of the class that implements the FrameProcessor interface
	void setFrameProcessor(FrameProcessor* frameProcessorPtr) {

		// invalidate callback function
		process = 0;

		frameProcessor = frameProcessorPtr;
		callProcess();
	}

	// stop streaming at this frame number
	void stopAtFrameNo(long frame) {

		frameToStop = frame;
	}

	// process callback to be called
	void callProcess() {

		callIt = true;
	}

	// do not call process callback
	void dontCallProcess() {

		callIt = false;
	}

	// to display the input frames
	void displayInput(std::string wn) {

		windowNameInput = wn;
		cv::namedWindow(windowNameInput);
	}

	// to display the processed frames
	void displayOutput(std::string wn) {

		windowNameOutput = wn;
		cv::namedWindow(windowNameOutput);
	}

	// do not display the processed frames
	void dontDisplay() {

		cv::destroyWindow(windowNameInput);
		cv::destroyWindow(windowNameOutput);
		windowNameInput.clear();
		windowNameOutput.clear();
	}

	// set a delay between each frame, 0 means wait at each frame, negative means no delay
	void setDelay(int d) {

		delay = d;
	}

	// a count is kept of the processed frames
	long getNumberOfprocessedFrames() {

		return fnumber;
	}

	// return the size of the video frame
	cv::Size getFrameSize() {

		if (images.size() == 0) {

			// get size of from the capture device
			int w = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
			int h = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));

			return cv::Size(w, h);

		}
		else {    // if input is vector of images

			cv::Mat tmp = cv::imread(images[0]);
			if (!tmp.data) return cv::Size(0, 0);
			else return tmp.size();
		}
	}

	// return the frame number of the next frame
	long getFrameNumber() {

		if (images.size() == 0) {

			// get info of from the capture device
			long f = static_cast<long>(capture.get(cv::CAP_PROP_POS_FRAMES));
			return f;

		}
		else {

			return static_cast<long>(itImg - images.begin());
		}
	}

	// return the position in ms
	double getPositionMS() {

		// undefined for vector of images
		if (images.size() != 0) return 0.0;

		double t = capture.get(cv::CAP_PROP_POS_MSEC);
		return t;
	}

	// return the frame rate
	double getFrameRate() {

		// undefined for vector of images
		if (images.size() != 0) return 0;

		double r = capture.get(cv::CAP_PROP_FPS);
		return r;
	}

	// return the number of frames in video
	long getTotalFrameCount() {

		// for vector of images
		if (images.size() != 0) return images.size();

		long t = capture.get(cv::CAP_PROP_FRAME_COUNT);
		return t;
	}

	// get the codec of input video
	int getCodec(char codec[4]) {

		// undefined for vector of images
		if (images.size() != 0) return -1;

		union {
			int value;
			char code[4];
		} returned;

		returned.value = static_cast<int>(capture.get(cv::CAP_PROP_FOURCC));

		codec[0] = returned.code[0];
		codec[1] = returned.code[1];
		codec[2] = returned.code[2];
		codec[3] = returned.code[3];

		return returned.value;
	}

	// go to this frame number
	bool setFrameNumber(long pos) {

		// for vector of images
		if (images.size() != 0) {

			// move to position in vector
			itImg = images.begin() + pos;
			// is it a valid position?
			if (pos < images.size())
				return true;
			else
				return false;

		}
		else { // if input is a capture device

			return capture.set(cv::CAP_PROP_POS_FRAMES, pos);
		}
	}

	// go to this position
	bool setPositionMS(double pos) {

		// not defined in vector of images
		if (images.size() != 0)
			return false;
		else
			return capture.set(cv::CAP_PROP_POS_MSEC, pos);
	}

	// go to this posion expressed in fraction of total film length
	bool setRelativePosition(double pos) {

		// for vector of images
		if (images.size() != 0) {

			// move to position in vector
			long posI = static_cast<long>(pos * images.size() + 0.5);
			itImg = images.begin() + posI;
			// is it a valid position?
			if (posI < images.size()) return true;
			else return false;

		}
		else {

			return capture.set(cv::CAP_PROP_POS_AVI_RATIO, pos);
		}
	}

	// Stop the processing
	void stopIt() {

		stop = true;
	}

	// is the process stopped?
	bool isStopped() {

		return stop;
	}

	// is the capture device opened?
	bool isOpened() {

		return capture.isOpened() || !images.empty();
	}

	// to grab (and process) the frames of the sequence
	void run() {

		// current frame
		cv::Mat frame;
		// output frame
		cv::Mat output;

		// if no capture device has been set
		if (!isOpened()) return;

		stop = false;

		while (!isStopped()) {

			// read next frame if any
			if (!readNextFrame(frame)) break;

			// display input frame
			if (windowNameInput.length() != 0)
				cv::imshow(windowNameInput, frame);

			// calling the process function or method
			if (callIt) {

				// process the frame
				if (process)
					process(frame, output);
				else if (frameProcessor)
					frameProcessor->process(frame, output);

				// increment frame number
				fnumber++;

			}
			else {

				output = frame;
			}

			// write output sequence
			if (outputFile.length() != 0)
				writeNextFrame(output);

			// display output frame
			if (windowNameOutput.length() != 0)
				cv::imshow(windowNameOutput, output);

			// introduce a delay
			if (delay >= 0 && cv::waitKey(delay) >= 0)
				stopIt();

			// check if we should stop
			if (frameToStop >= 0 && getFrameNumber() == frameToStop)
				stopIt();
		}
	}

};

#endif
