#include <time.h>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include "CycleTimer.h"

using namespace cv;

// parameter
int winSize     = 7;
int searchRange = 100;
float scale     = 1;

// auto parameter
int halfWinSize     = winSize /  2;
int halfSearchRange = searchRange / 2;

Mat1f getPatch(const Mat1f &img, int cx, int cy) {
	Range rangeY(cy - halfWinSize, cy + halfWinSize + 1);
	Range rangeX(cx - halfWinSize, cx + halfWinSize + 1);
	return img(rangeY, rangeX);
}

void stereoMatch(const Mat1f &imgSrc, const Mat1f &imgDst, Mat1f &disparity) {
	disparity = Mat1f::zeros( imgSrc.size() );

	
  for (int cy = halfWinSize; cy < imgSrc.rows-halfWinSize; ++cy) {
		for (int cxSrc = halfWinSize; cxSrc < imgSrc.cols-halfWinSize; ++cxSrc) {
			// left patch
			Mat1f patchL = getPatch( imgSrc, cxSrc, cy );

			// epipolar line search range
			int cxDstMin = max(cxSrc - halfSearchRange, halfWinSize);
			int cxDstMax = min(cxSrc + halfSearchRange, imgSrc.cols-halfWinSize);

			// find best match disparity
			float minSad        = FLT_MAX;
			float bestDisparity = 0;
			#pragma omp parallel for num_threads(8)
      for (int cxDst = cxDstMin; cxDst < cxDstMax; ++cxDst) {
				// right patch
				Mat1f patchR = getPatch( imgDst, cxDst, cy );

				// patch diff: Sum of Absolute Difference
				Mat1f diff = abs( patchL - patchR );
				float sad  = sum(diff)[0];

				// update best SAD and disparity
				if ( sad < minSad ) {
					minSad        = sad;
					bestDisparity = abs( cxDst - cxSrc );
				}
			}

			disparity(cy, cxSrc) = bestDisparity;
		}
	}
}

int main(int argc, char *argv[]) {
	const char *fileNameL = argv[1];
	const char *fileNameR = argv[2];

	// read input image
	Mat1f imgL, imgR;
	imread(fileNameL, 0).convertTo(imgL, IMREAD_GRAYSCALE);
	imread(fileNameR, 0).convertTo(imgR, IMREAD_GRAYSCALE);

	// scaling down
	resize(imgL, imgL, Size(), scale, scale);
	resize(imgR, imgR, Size(), scale, scale);

	// compute disparity
	double startTime = CycleTimer::currentSeconds();
	Mat1f disparityL, disparityR;
	stereoMatch(imgL, imgR, disparityL); // disparity Left
	stereoMatch(imgR, imgL, disparityR); // disparity Right
	double endTime = CycleTimer::currentSeconds();
	printf("time: %f sec\n", (double)  (endTime-startTime));

	// write file
	normalize(disparityL, disparityL, 0, 1, NORM_MINMAX);
	normalize(disparityR, disparityR, 0, 1, NORM_MINMAX);
	imshow("disparityL", disparityL);
	imshow("disparityR", disparityR);
	imwrite("disparityL.png", disparityL*255);
	imwrite("disparityR.png", disparityR*255);
	waitKey();

	return 0;
}