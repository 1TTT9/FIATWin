#include "stdafx.h"
#include "Utils.h"
#include <fstream>

#pragma region Path Functions
std::string getexepath()
{
	char the_path[256];
#ifdef _WINDOWS
	_getcwd(the_path, 255);
	strcat_s(the_path, "\\");
#else
	getcwd(the_path, 255);
	strcat(the_path, "/");
#endif
	return string(the_path);
}

std::string getAbsolutePath(string path) {
	char actualpath[PATH_MAX];

#ifdef _WINDOWS
	_fullpath(actualpath, path.c_str(), PATH_MAX);
#else
	char *ptr = realpath(path.c_str(), actualpath);
#endif
	return std::string(actualpath);
}

string getbase(string csv_filename) {
	std::string base_filename = string(csv_filename).substr(0, string(csv_filename).find_last_of("/\\") + 1);
	if (base_filename.length() == 0)
		base_filename = getexepath();
	// cout << "exe path " << getexepath() << endl;
	return base_filename;
}

string getbasename(string csv_filename) {
	return csv_filename.substr(csv_filename.find_last_of("/\\") + 1, csv_filename.length());
}
#pragma endregion Path Functions


#pragma region Processing
//process image
void extractRect(Mat image, Rect & plateZone, vector<vector<cv::Point> > & all_contours, vector<vector<cv::Point> > & orderedContours, vector<cv::Rect> & orderedRects) {

	vector<vector<cv::Point> > contours_inside;
	vector<cv::Rect> boundRects_inside;
	for (int i = 0; i < all_contours.size(); i++) {
		// filter on small contours
		if (contourArea(all_contours[i]) > image.rows * image.cols / 1000000) {
			cv::Rect bR = boundingRect(all_contours[i]);
			//filter countours in plate zone if platezone exists
			if (inside(bR, plateZone)) {
				contours_inside.push_back(all_contours[i]);
				boundRects_inside.push_back(bR);
			}
		}
	}
	cout << "      Nb big enough contours inside zone : " << contours_inside.size() << endl;

	// delete contours inside other
	vector<vector<cv::Point> > contours;
	vector<cv::Rect> boundRects;
	for (int i = 0; i < contours_inside.size(); i++) {

		bool sup = true;
		for (int j = 0; j < contours_inside.size(); j++)
			if (boundRects_inside[j].contains(boundRects_inside[i].tl()) && boundRects_inside[j].contains(boundRects_inside[i].br())) {
				sup = false;
				break;
			}
		if (sup) {
			contours.push_back(contours_inside[i]);
			boundRects.push_back(boundRects_inside[i]);
			//getOrientation( contours_inside[i], image.image);
		}
	}
	cout << "      Nb big enough and deduplicated contours inside zone : " << contours.size() << endl;
	if (contours.size() == 0) return;


	// distance de couleur entre les lettres
	vector<vector<cv::Point> > contours2;
	vector<cv::Rect> boundRects2;
	Mat mask = Mat::zeros(image.rows, image.cols, CV_8UC1);
	drawContours(mask, contours, -1, Scalar(255), CV_FILLED);
	//imshow("mask", mask);
	int channels[] = { 0,1,2 };
	float sranges[] = { 0, 256 };
	const float* ranges[] = { sranges, sranges, sranges };
	int histSize[] = { 25, 25, 25 };

	Mat b_hist;
	calcHist(&image, 1, channels, mask, b_hist, 3, histSize, ranges, true, false);
	normalize(b_hist, b_hist);

	for (int j = 0; j < contours.size(); j++) {
		Mat sub = image(boundRects[j]);
		Mat sub_mask = mask(boundRects[j]);
		// imshow("mask", sub_mask);
		Mat b_hist_j;
		calcHist(&sub, 1, channels, sub_mask, b_hist_j, 3, histSize, ranges, true, false);
		normalize(b_hist_j, b_hist_j);
		float distance = 0;
		for (int x = 0; x < histSize[0]; x++)
			for (int y = 0; y < histSize[1]; y++)
				for (int z = 0; z < histSize[2]; z++)
				{
					distance += min(b_hist_j.at<float>(x, y, z), b_hist.at<float>(x, y, z));
				}
		//cout << distance << endl;
		if (distance > 1.7) //1.70
		{
			contours2.push_back(contours[j]);
			boundRects2.push_back(boundRects[j]);
			getOrientation(contours[j], image);
		}
	}

	//order RECTS
	int min_rect_abs = -1;
	for (int j = 0; j < boundRects2.size(); j++) {
		int argmin = 0;
		int current_max = 1000000000;
		for (int i = 0; i < boundRects2.size(); i++)
			if ((boundRects2[i].tl().x < current_max) && (boundRects2[i].tl().x > min_rect_abs)) {
				argmin = i;
				current_max = boundRects2[argmin].tl().x;
			}
		orderedRects.push_back(boundRects2[argmin]);
		orderedContours.push_back(contours2[argmin]);
		min_rect_abs = boundRects2[argmin].tl().x;
	}

}

Mat resizeContains(Mat image, int cols, int rows, int & left, int & right, int & up, int & down, bool noise) {

	//resize
	Mat resized_image;
	double normalization_factor_width = ((float)cols) / ((float)image.size().width);
	double normalization_factor_height = ((float)rows) / ((float)image.size().height);
	double normalization_factor = std::min(normalization_factor_width, normalization_factor_height);
	resize(image, resized_image, Size(), normalization_factor, normalization_factor, INTER_LINEAR);

	//makeborder
	RNG rng;
	Mat last_image;
	if (noise)
		add_salt_pepper_noise(last_image, 0.3, 0.3, &rng);

	left = round((cols - resized_image.size().width) / 2.0);
	right = cols - resized_image.size().width - left;
	up = round((rows - resized_image.size().height) / 2.0);
	down = rows - resized_image.size().height - up;
	if (noise)
		copyMakeBorder(resized_image, last_image, up, down, left, right, BORDER_TRANSPARENT);
	else
		copyMakeBorder(resized_image, last_image, up, down, left, right, BORDER_CONSTANT, white);


	resized_image.release();
	return last_image;
}

void detectRectsAndContours(CascadeClassifier * cc, WorkImage image, vector<cv::Rect> & plateZone, vector<vector<cv::Point> > & all_contours) {
	//PLATE DETECTION
	cc->detectMultiScale(image.gray_image, plateZone, 1.05, 5);
	cout << "Nb plate zones : " << plateZone.size() << endl;

	if (plateZone.size() == 0)
		return;

	//LETTER DETECTION
	vector<Vec4i> hierarchy;
	findContours(image.threshold_image.clone(), all_contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	cout << "Nb contours found : " << all_contours.size() << endl;

	// Mat canny_output;
	// int thresh = 100;
	// int max_thresh = 255;
	// RNG rng(12345);
	// Canny( image.threshold_image, canny_output, thresh, thresh*2, 3 );
	// findContours( canny_output, all_contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );
	// imwrite(output_dir + "/" +path.substr(path.find_last_of("/\\") + 1) , threshold_image );
}


#pragma endregion Processing

#pragma region Noise
// noise functions
void add_salt_pepper_noise(Mat &img, float pa, float pb, RNG *rng)
{
	Mat saltpepper_noise = Mat::zeros(img.rows, img.cols, CV_8U);
	randu(saltpepper_noise, 0, 255);

	Mat black, white;
	threshold(saltpepper_noise, black, 5.0, 255, cv::THRESH_BINARY_INV);
	threshold(saltpepper_noise, white, 250.0, 255, cv::THRESH_BINARY);

	img.setTo(255, white);
	img.setTo(0, black);
}

void add_gaussian_noise(Mat &srcArr, double mean, double sigma, RNG *rng)
{
	Mat NoiseArr = srcArr.clone();
	rng->fill(NoiseArr, RNG::NORMAL, mean, sigma);
	add(srcArr, NoiseArr, srcArr);
}
#pragma endregion Noise

#pragma region Display functions
void displayRects(Mat image, vector<cv::Rect> plateZone, Scalar color1) {
	for (int i = 0; i < plateZone.size(); i++) {
		rectangle(image, plateZone[i].tl(), plateZone[i].br(), color1, 2, 8, 0);
	}
}

void displayRectangle(Mat image, cv::Rect r, Scalar color1) {
	rectangle(image, r.tl(), r.br(), color1, 2, 8, 0);
}

void displayRotatedRectangle(Mat image, RotatedRect rRect, Scalar color1) {
	cv::Point2f vertices[4];
	rRect.points(vertices);
	for (int i = 0; i < 4; i++)
		line(image, vertices[i], vertices[(i + 1) % 4], color1, 2, 8, 0);
}

void displayCross(Mat image, Point2f p, Scalar color1) {
	line(image, p - Point2f(3.0, 0.0), p + Point2f(3.0, 0.0), color1, 1, 8, 0);
	line(image, p - Point2f(0.0, 3.0), p + Point2f(0.0, 3.0), color1, 1, 8, 0);
}

void displayText(Mat image, string text, cv::Point textOrg, double fontScale) {
	int fontFace = 0;

	int thickness = max((int)(3 * fontScale), 1);
	int baseline = 0;
	Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);
	baseline += thickness;
	rectangle(image, textOrg + Point(0, baseline + textSize.height / 2),
		textOrg + Point(textSize.width, -textSize.height / 2),
		Scalar(0, 0, 255), CV_FILLED);
	putText(image, text, textOrg + Point(0, textSize.height / 2), fontFace, fontScale, Scalar::all(255), thickness, (int)(8.0 * fontScale));
}
#pragma endregion Display Functions

#pragma region read image
Mat readI(string path) {
	Mat image = imread(path, CV_LOAD_IMAGE_COLOR);   // Read the file
	int height = image.rows, width = image.cols, area = height * width, channel = image.channels();
	double factor = image.rows / 500;
	if (factor > 1.5) {
		factor = 1 / factor;
		resize(image, image, Size(), factor, factor, INTER_AREA);
	}
	return image;
}

void readCSV(const char * csv_file, std::vector<std::vector<std::string> > & input) {

#ifdef _WINDOWS
	std::ifstream csv_f(csv_file);
#else
	std::ifstream csv_f(csv_file, std::ios::in | std::ios::binary);
#endif
	std::string str;

	while (std::getline(csv_f, str))
	{
		// Process str
		std::stringstream ss(str);
		std::vector<string> result;

		while (ss.good())
		{
			std::string substr;
			getline(ss, substr, ',');
			result.push_back(substr);
		}

		input.push_back(result);
	}

}

void group_by_image(std::vector<std::vector<std::string> > input, bool rotated, bool correct, float ratio, std::vector<std::string> & input_path, std::vector<std::vector<string> > &input_labels, std::vector<std::vector<RotatedRect> > &input_rotatedrects) {
	for (int i = 0; i < input.size(); i++) {
		// compute index
		int index = -1;
		for (int j = 0; j < input_path.size(); j++)
			if (input_path[j] == input[i][0])
				index = j;
		if (index == -1) {
			input_path.push_back(input[i][0]);
			std::vector<string> vec_int;
			std::vector<RotatedRect> vec_rotatedrect;
			input_rotatedrects.push_back(vec_rotatedrect);
			input_labels.push_back(vec_int);
			index = input_path.size() - 1;
		}
		// cout << "index : " << index << endl;

		// take orientation into account or not
		int orient = 0;
		if (rotated) orient = stoi(input[i][6]);
		// cout << "Orientation : " << orient << endl;
		Size2f s;
		if (correct)
			s = correct_ratio(stoi(input[i][4]), stoi(input[i][5]), ratio);
		else
			s = Size2f(stoi(input[i][4]), stoi(input[i][5]));
		input_rotatedrects[index].push_back(RotatedRect(Point2f(stoi(input[i][2]), stoi(input[i][3])), s, orient));

		input_labels[index].push_back(input[i][1]);
	}
}

void findRectangle(std::string image_path, std::string csvfile, vector<Rect> &outputRects) {
	std::ifstream file(csvfile);
	std::string str;
	while (std::getline(file, str))
	{
		std::stringstream ss(str);
		vector<string> result;

		while (ss.good())
		{
			string substr;
			getline(ss, substr, ',');
			result.push_back(substr);
		}
		if (result[0] == image_path)
			outputRects.push_back(Rect(stoi(result[2]) - stoi(result[4]) / 2.0, stoi(result[3]) - stoi(result[5]) / 2.0, stoi(result[4]), stoi(result[5])));
	}
}
#pragma endregion read image

#pragma region Rect

// rect functions

int getCenterX(cv::Rect r) {
	return r.tl().x + floor(((float)(r.br().x - r.tl().x)) / 2.0);
}

int getCenterY(cv::Rect r) {
	return  r.tl().y + floor(((float)(r.br().y - r.tl().y)) / 2.0);
}

bool inside(cv::Rect bR, cv::Rect plateZone) {
	int bx = bR.tl().x, by = bR.tl().y, bh = bR.size().height, bw = bR.size().width;
	//bR.area() > ((double) plateZone.area()) / 35
	return bR.size().height < plateZone.size().height && bR.size().height * 3 > plateZone.size().height && bR.size().width * 5 < plateZone.size().width && bR.size().width * 25 > plateZone.size().width && plateZone.contains(cv::Point2f(bx + bw / 2, by + bh / 2));
}

// compute if a rectangle is inside an image
bool is_in_image(Rect r, Mat img) {
	return ((r.tl().x >= 0) && (r.tl().y >= 0) && (r.br().x < img.cols) && (r.br().y < img.rows));
}
bool is_in_image(RotatedRect r, Mat img) {
	return is_in_image(r.boundingRect(), img);
}

Point2f change_ref(Point2f p, float center_x, float center_y, float orientation) {
	Point2f p_repositioned(p.x - center_x, p.y - center_y);
	double orientation_radian = orientation * 3.14159265 / 180.0;

	double hypothenuse = sqrt(p_repositioned.x * p_repositioned.x + p_repositioned.y * p_repositioned.y);
	double angle = atan2(p_repositioned.y, p_repositioned.x);
	return Point2f(hypothenuse * cos(angle - orientation_radian), hypothenuse * sin(angle - orientation_radian));
}


// compute if point is in rotated rectangle
bool is_in(Point p, RotatedRect rr) {
	Point2f p_new = change_ref(p, rr.center.x, rr.center.y, rr.angle);
	//cout << p_repositioned << endl;

	//cout << " compare " << (new_y < rr.size.height / 2.0)  << " - " << new_y << " - " << rr.size.height / 2.0 << endl;
	//cout << "is in " << ( (new_x  > - rr.size.width / 2.0) && (new_x  < rr.size.width / 2.0) && (new_y > - rr.size.height / 2.0) && (new_y < rr.size.height / 2.0)  ) ;
	return ((p_new.x >= -rr.size.width / 2.0) && (p_new.x <= rr.size.width / 2.0) && (p_new.y >= -rr.size.height / 2.0) && (p_new.y <= rr.size.height / 2.0));
}

// compute brute-force Intersection Over Union
float intersectionOverUnion(RotatedRect r1, RotatedRect r2) {
	Rect br1 = r1.boundingRect();
	Rect br2 = r2.boundingRect();
	Rect br(Point(min(br1.tl().x, br2.tl().x), min(br1.tl().y, br2.tl().y)), Point(max(br1.br().x, br2.br().x), max(br1.br().y, br2.br().y)));
	//cout << "Area : " << br.area() << endl;
	int intersection = 0;
	for (int i = 0; i <= br.size().width; i++)
		for (int j = 0; j <= br.size().height; j++) {
			Point p(br.tl().x + i, br.tl().y + j);
			if (is_in(p, r1) && is_in(p, r2))
				intersection++;
		}
	//cout << "BR : " << br2.tl() << " -> " << br2.br() << endl;
	//cout << "IOU " << ((float)intersection) << " - " << ((float) br.area()) << endl;
	return ((float)intersection) / ((float)br.area());
}
#pragma endregion Rect


#pragma region others 
// others
Mat createOne(vector<Mat> & images, int cols, int rows, int gap_size, int dim)
{
	cv::Mat result(rows * dim + (rows + 2) * gap_size, cols * dim + (cols + 2) * gap_size, images[0].type(), white);
	size_t i = 0;
	int current_height = gap_size;
	int current_width = gap_size;
	for (int y = 0; y < rows; y++) {
		for (int x = 0; x < cols; x++) {
			if (i >= images.size())
				return result;
			// get the ROI in our result-image
			cv::Mat to(result,
				cv::Range(current_height, current_height + dim),
				cv::Range(current_width, current_width + dim));
			// copy the current image to the ROI
			images[i++].copyTo(to);
			current_width += dim + gap_size;
		}
		// next line - reset width and update height
		current_width = gap_size;
		current_height += dim + gap_size;
	}
	return result;
}

void standard_deviation(vector<int> data, double & mean, double & stdeviation, double & median)
{
	std::sort(data.begin(), data.end());
	mean = 0.0;
	stdeviation = 0.0;
	int i;
	for (i = 0; i<data.size();++i)
		mean += data[i];
	mean = mean / data.size();
	for (i = 0; i< data.size();++i)
		stdeviation += (data[i] - mean)*(data[i] - mean);
	stdeviation = sqrt(stdeviation / data.size());
	median = round(data.size() / 2);
	return;
}

double getOrientation(vector<cv::Point> &pts, Mat &img)
{
	//Construct a buffer used by the pca analysis
	Mat data_pts = Mat(pts.size(), 2, CV_64FC1);
	for (int i = 0; i < data_pts.rows; ++i)
	{
		data_pts.at<double>(i, 0) = pts[i].x;
		data_pts.at<double>(i, 1) = pts[i].y;
	}

	//Perform PCA analysis
	PCA pca_analysis(data_pts, Mat(), CV_PCA_DATA_AS_ROW);

	//Store the position of the object
	cv::Point pos = cv::Point(pca_analysis.mean.at<double>(0, 0),
		pca_analysis.mean.at<double>(0, 1));

	//Store the eigenvalues and eigenvectors
	vector<cv::Point2d> eigen_vecs(2);
	vector<double> eigen_val(2);
	for (int i = 0; i < 2; ++i)
	{
		eigen_vecs[i] = cv::Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
			pca_analysis.eigenvectors.at<double>(i, 1));

		eigen_val[i] = pca_analysis.eigenvalues.at<double>(0, i);
	}

	// Draw the principal components
	circle(img, pos, 3, CV_RGB(255, 0, 255), 2);
	if (eigen_vecs[0].x * eigen_val[0] * eigen_vecs[0].x * eigen_val[0] + eigen_vecs[0].y * eigen_val[0] * eigen_vecs[0].y * eigen_val[0] > eigen_vecs[1].x * eigen_val[1] * eigen_vecs[1].x * eigen_val[1] + eigen_vecs[1].y * eigen_val[1] * eigen_vecs[1].y * eigen_val[1]) {
		line(img, pos, pos + 0.2 * cv::Point(eigen_vecs[0].x * eigen_val[0], eigen_vecs[0].y * eigen_val[0]), CV_RGB(255, 255, 0));
		return atan2(eigen_vecs[0].y, eigen_vecs[0].x) * 180 / 3.1417 - 90;
	}
	else {
		line(img, pos, pos + 0.2 * cv::Point(eigen_vecs[1].x * eigen_val[1], eigen_vecs[1].y * eigen_val[1]), CV_RGB(0, 255, 255));
		return atan2(eigen_vecs[1].y, eigen_vecs[1].x) * 180 / 3.1417 - 90;
	}

}

void myGetQuadrangleSubPix(const Mat& src, Mat& dst, Mat& m)
{

	cv::Size win_size = dst.size();
	double matrix[6];
	cv::Mat M(2, 3, CV_64F, matrix);
	m.convertTo(M, CV_64F);
	double dx = (win_size.width - 1)*0.5;
	double dy = (win_size.height - 1)*0.5;
	matrix[2] -= matrix[0] * dx + matrix[1] * dy;
	matrix[5] -= matrix[3] * dx + matrix[4] * dy;

	// RNG rng;
	// add_salt_pepper_noise(dst,0.3,0.3,&rng);
	cv::warpAffine(src, dst, M, dst.size(),
		cv::INTER_LINEAR + cv::WARP_INVERSE_MAP,
		cv::BORDER_CONSTANT);
}

void getRotRectImg(cv::RotatedRect rr, Mat &img, Mat& dst)
{
	Mat m(2, 3, CV_64FC1);
	float ang = rr.angle*CV_PI / 180.0;
	m.at<double>(0, 0) = cos(ang);
	m.at<double>(1, 0) = sin(ang);
	m.at<double>(0, 1) = -sin(ang);
	m.at<double>(1, 1) = cos(ang);
	m.at<double>(0, 2) = rr.center.x;
	m.at<double>(1, 2) = rr.center.y;
	myGetQuadrangleSubPix(img, dst, m);
}

Mat extractRotatedRect(Mat src, RotatedRect rect) {
	Mat dst(rect.size, CV_32FC3);
	getRotRectImg(rect, src, dst);
	return dst;
	// // matrices we'll use
	// Mat M, rotated, cropped;
	// // get angle and size from the bounding box
	// float angle = rect.angle;
	// Size rect_size = rect.size;
	// // thanks to http://felix.abecassis.me/2011/10/opencv-rotation-deskewing/
	// if (rect.angle < -45.) {
	//     angle += 90.0;
	//     std::swap(rect_size.width, rect_size.height);
	// }
	// // get the rotation matrix
	// M = getRotationMatrix2D(rect.center, angle, 1.0);
	// // perform the affine transformation
	// warpAffine(src, rotated, M, src.size(), INTER_CUBIC,cv::BORDER_CONSTANT);
	// // crop the resulting image
	// getRectSubPix(rotated, rect_size, rect.center, cropped);
	// return cropped;
}

void correct_ratio(Rect & r, double ratio) {
	float width = (float)r.size().width;
	float height = (float)r.size().height;

	float current_ratio = height / width;

	if (current_ratio > ratio) {
		// augment width
		int new_width_delta = floor((height / ratio - width) / 2.0);

		r.x -= new_width_delta;
		r.width += new_width_delta * 2;

	}
	else {
		// augment height
		int new_height_delta = floor((width * ratio - height) / 2.0);

		r.y -= new_height_delta;
		r.height += new_height_delta * 2;


	}
};

Size2f correct_ratio(float width, float height, double ratio) {

	float current_ratio = height / width;

	if (current_ratio > ratio) {
		// augment width
		int new_width = floor(height / ratio);

		return Size(new_width, height);
		// r.size.width = new_width;

	}
	else {
		// augment height
		int new_height = floor(width * ratio);

		return Size(width, new_height);
		// r.size.height = new_height;

	}
};

bool PairCompare(const std::pair<float, int>& lhs,
	const std::pair<float, int>& rhs) {
	return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
std::vector<int> Argmax(const std::vector<float>& v, int N) {
	std::vector<std::pair<float, int> > pairs;
	for (size_t i = 0; i < v.size(); ++i)
		pairs.push_back(std::make_pair(v[i], i));

	unsigned long max_N = std::min<unsigned long>((unsigned long)N, pairs.size());
	std::partial_sort(pairs.begin(), pairs.begin() + max_N, pairs.end(), PairCompare);

	std::vector<int> result;
	for (int i = 0; i < max_N; ++i)
		result.push_back(pairs[i].second);
	return result;
}
#pragma endregion others