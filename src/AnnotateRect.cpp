#include "stdafx.h"
#include <io.h>
#include <string>
#define NOMINMAX
#include "AnnotateRect.h"

#include <time.h>
#include <fstream>
#include "WorkImage.h"

#include <glog/logging.h>

using std::min;
using std::max;

struct couple {
	string name;
	Mat image;
	// current rectangle
	RotatedRect r;
	int border_right;
	int border_left;
	int border_top;
	int border_bottom;
	bool init;
	bool cross;
	// list of rectangles to annotate
	std::vector<RotatedRect> init_rectangles;
	//list of annotated rectangles
	std::vector<RotatedRect> rectangles;
	int mode;
	float image_factor;
	float ratio;
};


float height;
float width;
float orientation;
float scale_factor = 1.07;
float fast_scale_factor = 1.5;
int orientation_step = 2;
int fast_orientation_step = 20;
int position_step = 1;
int fast_position_step = 20;

int digit_number = 1;
int digit_number_count = 0;


void displayRR(couple &cc, string export_dir = "");

// mouse click
void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	couple* cc = (couple*)userdata;
	
	if (flags == EVENT_FLAG_CTRLKEY)
	{
		//cout << "pressing CTRL key - position (" << x << ", " << y << ")" << endl;
	}
	else if (flags == EVENT_FLAG_SHIFTKEY)
	{
		//cout << "pressing Shift key - position (" << x << ", " << y << ")" << endl;
	}

	if (event == EVENT_LBUTTONDOWN)
	{
		//cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
		if ((*cc).init) {
			// use last clic to compute from diagonal
			// diagonal ((*cc).r.center.x, (*cc).r.center.y) -> Point2f((float)x,(float)y)
			Point2f p1((*cc).r.center.x, (*cc).r.center.y);
			Point2f p2((float)x, (float)y);
			Point2f p_center((p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0);
			float hyp_2 = (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y);
			width = sqrt(hyp_2 / (1.0 + pow((*cc).ratio, 2.0)));
			height = width * (*cc).ratio;
			orientation = (atan2(p2.y - p1.y, p2.x - p1.x) - asin(height / sqrt(hyp_2))) * 180.0 / 3.1415; //
			(*cc).r = RotatedRect(p_center, Size2f(width, height), orientation);
		}
		else {
			(*cc).r = RotatedRect(Point2f((float)x, (float)y), Size2f(width, height), orientation);
			(*cc).init = true;
		}
		displayRR((*cc));
	}
}

// display image and rectangles
void displayRR(couple& cc, string export_dir) {
	Mat copy_image;
	copyMakeBorder(cc.image, copy_image, cc.border_top, cc.border_bottom, cc.border_left, cc.border_right, BORDER_CONSTANT, green);
	cc.r.size = Size2f(width, height);
	cc.r.angle = orientation;

	if (!cc.init)
		if (cc.init_rectangles.size() > 0) {
			cc.r = cc.init_rectangles[0];
			cc.init_rectangles.erase(cc.init_rectangles.begin());
			cc.init = true;
		}

	//display rectangles
	for (int i = 0; i < cc.rectangles.size(); i++) {
		RotatedRect ddd(cc.rectangles[i].center + Point2f(cc.border_left, cc.border_top), cc.rectangles[i].size, cc.rectangles[i].angle);
		if (cc.cross)
			displayCross(copy_image, ddd.center, yellow);
		else {
			displayRotatedRectangle(copy_image, ddd, yellow);
			arrowedLine(copy_image, ddd.center, ddd.center + Point2f(100.0 * cos((ddd.angle - 90) * 3.141516 / 180.0), 100.0 * sin((ddd.angle - 90) * 3.141516 / 180.0)), Scalar(0, 0, 0));
		}
	}

	//display init rectangles
	for (int i = 0; i < cc.init_rectangles.size(); i++) {
		RotatedRect ddd(cc.init_rectangles[i].center + Point2f(cc.border_left, cc.border_top), cc.init_rectangles[i].size, cc.init_rectangles[i].angle);
		if (cc.cross)
			displayCross(copy_image, ddd.center, green);
		else {
			displayRotatedRectangle(copy_image, ddd, green);
			arrowedLine(copy_image, ddd.center, ddd.center + Point2f(100.0 * cos((ddd.angle - 90) * 3.141516 / 180.0), 100.0 * sin((ddd.angle - 90) * 3.141516 / 180.0)), Scalar(0, 0, 0));
		}
	}

	// display current rectangle
	if (cc.init) {
		//cout << "Display rectangle " << cc.r.center.x << "," << cc.r.center.y << "," << cc.r.size.width << "," << cc.r.size.height << endl;
		RotatedRect rrrr = RotatedRect(cc.r.center + Point2f(cc.border_left, cc.border_top), cc.r.size, cc.r.angle);
		displayRotatedRectangle(copy_image, rrrr, blue);
		arrowedLine(copy_image, rrrr.center, rrrr.center + Point2f(100.0 * cos((rrrr.angle - 90) * 3.141516 / 180.0), 100.0 * sin((rrrr.angle - 90) * 3.141516 / 180.0)), Scalar(0, 0, 0));
	}

	if (export_dir == "") {
		namedWindow("DisplayW", WINDOW_AUTOSIZE);
		setMouseCallback("DisplayW", CallBackFunc, &cc);
		imshow("DisplayW", copy_image);
	}
	else
		imwrite(export_dir + "/" + cc.name, copy_image);

}

//save the annotated rectangle
void saveRR(couple& cc, string image_path, float factor, char classe, std::ofstream *outfile) {//image, outfile
	if (cc.r.center.x != 0) {
		int new_x = (int)(((float)cc.r.center.x - cc.border_left) / factor);
		int new_y = (int)(((float)cc.r.center.y - cc.border_top) / factor);
		int new_w = (int)(((float)cc.r.size.width) / factor);
		int new_h = (int)(((float)cc.r.size.height) / factor);
		//cout << "Save " << classe << "," << new_x << ":" << new_y << ":" << new_w << ":" << new_h << endl;
		LOG(INFO) << "Save " << classe << "," << new_x << ":" << new_y << ":" << new_w << ":" << new_h;
		*outfile << image_path << "," << classe << "," << new_x << "," << new_y << "," << new_w << "," << new_h << "," << cc.r.angle << endl;
		outfile->flush();
		cc.rectangles.push_back(RotatedRect(Point(new_x * factor, new_y * factor), Size(new_w * factor, new_h * factor), cc.r.angle));
		cc.init = false;
		displayRR(cc);
	}
	else 
		LOG(ERROR) << "Rectangle center not initialized";
}

//annotation process
int annotate(char * input_dir, char* csv_file, float ratio, string init_rectangles_file, bool cross, string export_dir, int _digit_number)
{
	int mode = 0;

	digit_number = _digit_number;

	time_t last_timer;
	time_t new_timer;
#ifdef _WINDOWS
	time(&new_timer);
#endif
	vector<int> times(10, 0);

	// read input csv if exists
	std::vector<std::vector<std::string> > input;
	readCSV(csv_file, input);
	string csv_dir_path = string(csv_file).substr(0, string(csv_file).find_last_of("/\\") + 1);
	LOG(INFO) << "CSV dir path : " << csv_dir_path;

	// read the init rectangles
	bool rotated = false;
	bool correct = false;
	std::vector<std::string> init_rectangles_path;
	std::vector<std::vector<string> > init_rectangles_labels;
	std::vector<std::vector<RotatedRect> > init_rectangles;
	string init_csv_dir_path = "";
	if (init_rectangles_file != "") {
		init_csv_dir_path = init_rectangles_file.substr(0, init_rectangles_file.find_last_of("/\\") + 1);
		LOG(INFO) << "Init CSV dir path : " << init_csv_dir_path;

		std::vector<std::vector<std::string> > init_rectangles_boxes;
		readCSV(init_rectangles_file.c_str(), init_rectangles_boxes);
		group_by_image(init_rectangles_boxes, rotated, correct, ratio, init_rectangles_path, init_rectangles_labels, init_rectangles);
	}

	std::ofstream outfile;
	outfile.open(std::string(csv_file), std::ios_base::app);

	struct dirent *entry;
	int ret = 1;
	DIR *dir;
	dir = opendir(input_dir);
	string dir_name = string(input_dir).substr(string(input_dir).find_last_of("/\\") + 1);
	LOG(INFO) << "data dir path : " << dir_name;

	DIR *dp;
	int numOfImages = 0;
	struct dirent *ep;
	dp = opendir(input_dir);
	if (dp != NULL)
	{
		while (ep = readdir(dp))
		{
			if (ep->d_namlen <= 4)
				continue;
			string _name(ep->d_name);
			_name = _name.substr(ep->d_namlen - 3, 3);
			if (_name == "jpg" || _name == "png")
			{
				numOfImages++;
				LOG(INFO) << numOfImages << ":" << _name << "(" << ep->d_namlen;
			}
		}
		(void)closedir(dp);
	}
	else
		LOG(ERROR) << "Couldn't open the directory: " << input_dir;
	LOG(INFO) << "Number of files: " << numOfImages;

	// ratio = atof(argv[3]);
	height = ratio * 100.0;
	width = 100.0;
	orientation = 0.0;
	char classe = '0';

	bool isBrokenUp = false;
	while ((entry = readdir(dir)) != NULL && !isBrokenUp) {
		string image_path = std::string(input_dir) + "/" + std::string(entry->d_name);
		LOG(INFO) << "Add rectangle on " << entry->d_name << ", path: "<< image_path;
		// string relative_image_path = dir_name + "/" + std::string(entry->d_name);
		WorkImage image(image_path);
		if (image.ok)
		{
			couple cc;
			cc.ratio = ratio;
			cc.name = std::string(entry->d_name);
			cc.image_factor = image.factor;
			LOG(INFO) << "Image factor: " << image.factor;
			cc.cross = cross;
			cc.mode = 1;
			cc.init = false;
			cc.image = image.image;
			cc.border_top = 0;
			cc.border_bottom = 0;
			cc.border_right = 0;
			cc.border_left = 0;

			// compute the max bounding box
			Rect max_bounding_box(Point(1, 1), Size(image.image.cols, image.image.rows) + Size(-2, -2));

			// add previously annotated rectangles
			for (int i = 0; i < input.size(); i++)
				if (getAbsolutePath(csv_dir_path + input[i][0]) == getAbsolutePath(image_path)) {
					RotatedRect rrd(Point2f(stof(input[i][2]) * image.factor, stof(input[i][3])* image.factor), Size2f(stof(input[i][4])* image.factor, stof(input[i][5])* image.factor), stof(input[i][6]));
					max_bounding_box = max_bounding_box | rrd.boundingRect();
					cc.rectangles.push_back(rrd);
				}
			LOG(INFO) << "Previously annotated rectangles : " << cc.rectangles.size();
			// work with init rectangles
			int index = -1;
			// find the group of init rectangles of the images
			for (int i = 0; i < init_rectangles_path.size(); i++) {
				if (getAbsolutePath(init_csv_dir_path + init_rectangles_path[i]) == getAbsolutePath(image_path))
					index = i;
			}
			// add init rectangles that have not been already added
			if (index != -1) {
				for (int i = 0; i < init_rectangles[index].size(); i++) {
					bool seen = false;
					RotatedRect rrr(Point2f(init_rectangles[index][i].center.x * image.factor, init_rectangles[index][i].center.y * image.factor), Size2f(init_rectangles[index][i].size.width * image.factor, init_rectangles[index][i].size.height * image.factor), init_rectangles[index][i].angle);
					max_bounding_box = max_bounding_box | rrr.boundingRect();
					for (int j = 0; j < cc.rectangles.size(); j++)
						if (cc.rectangles[j].center.x == rrr.center.x  && cc.rectangles[j].center.y == rrr.center.y && cc.rectangles[j].size.width == rrr.size.width && cc.rectangles[j].size.height == rrr.size.height)
							seen = true;
					if (!seen)
						cc.init_rectangles.push_back(rrr);
				}
				LOG(INFO) << "Init rectangles : " << cc.init_rectangles.size();
			}
			//cout << "max bounding box " << max_bounding_box << endl;
			//cout << "Image dim " << image.image.cols << "," << image.image.rows << endl;
			cc.border_left = -std::min(max_bounding_box.x - 1, 0);
			cc.border_right = max(max_bounding_box.x + max_bounding_box.width + 1, image.image.cols) - image.image.cols;
			cc.border_top = -std::min(max_bounding_box.y - 1, 0);
			cc.border_bottom = max(max_bounding_box.y + max_bounding_box.height + 1, image.image.rows) - image.image.rows;
			LOG(INFO) << "Margins : " << cc.border_top << ", " << cc.border_right << ", " << cc.border_bottom << ", " << cc.border_left;

			displayRR(cc, export_dir);
			if (export_dir != "")
				continue;

			int k;
			while (1) {
				k = waitKeyEx(0);
				cout << "key : " << k << endl;
				last_timer = new_timer;
#ifdef _WINDOWS
				time(&new_timer);
#else
				new_timer = time(NULL);
#endif
				int seconds = difftime(new_timer, last_timer);
				//cout << "Time : " << seconds << endl;
				std::vector<int>::iterator it = times.begin();
				it = times.insert(it, seconds);
				times.pop_back();

				int sum = 0;
				for (int i = 0; i < times.size(); i++) {
					sum += times[i];
					//cout << times[i] << ",";
				}
				//cout << endl << "Times : " << sum << endl;
				// if( k == '\t') {
				//   cout << "Arrow keys for " << (cc.mode?"Fast":"Slow") << endl;
				//   cc.mode = cc.mode?0:1;
				//   displayRR(cc);
				//
				// } else
				
				if (k == 32 /*sapcer*/) {
					//space
					//cout << "Change mode to ";
					if (mode) {
						mode = 0;
						//cout << "Position" << endl;
					}
					else {
						//cout << "Rotation/Scale" << endl;
						mode = 1;
					}
				} 	
				else if (k == 555 /*ctrl+s,  enlarge width*/) {
					if (cc.init) {
						if (sum == 0) {
							width *= fast_scale_factor;
							//cout << "New size : " << int(width) << "," << int(height) << endl;
						}
						else {
							width *= scale_factor;
							//cout << "New size : " << int(width) << "," << int(height) << endl;
						}
						displayRR(cc);
					}
				}
				else if (k == 777 /*ctrl+w,  enlarge height*/) {
					if (cc.init) {
						if (sum == 0) {
							height *= fast_scale_factor;
							//cout << "New size : " << int(width) << "," << int(height) << endl;
						}
						else {
							height *= scale_factor;
							//cout << "New size : " << int(width) << "," << int(height) << endl;
						}
						displayRR(cc);
					}
				}
				else if (k == 666 /*ctrl+x, downscale width*/) {
					//cout << "FN+ Down" << endl;
					if (cc.init) {
						if (sum == 0) {
							width /= fast_scale_factor;
							//cout << "New size : " << int(width) << "," << int(height) << endl;
						}
						else {
							width /= scale_factor;
							//cout << "New size : " << int(width) << "," << int(height) << endl;
						}
						displayRR(cc);
					}
				}
				else if (k == 888 /*ctrl+x, downscale height*/) {
					if (cc.init) {
						if (sum == 0) {
							height /= fast_scale_factor;
							//cout << "New size : " << int(width) << "," << int(height) << endl;
						}
						else {
							height /= scale_factor;
							//cout << "New size : " << int(width) << "," << int(height) << endl;
						}
						displayRR(cc);
					}
				}
				else if (k == 23 /*ctrl+w,  enlarge size*/) {
					//cout << "FN+ Up" << endl;
					if (cc.init) {
						if (sum == 0) {
							height *= fast_scale_factor;
							width *= fast_scale_factor;
							//cout << "New size : " << int(width) << "," << int(height) << endl;
						}
						else {
							height *= scale_factor;
							width *= scale_factor;
							//cout << "New size : " << int(width) << "," << int(height) << endl;
						}

						displayRR(cc);
					}
				}
				else if (k == 2490368 /*up-arrow, shift up*/) {

					//cout << "Up" << endl;
					if (cc.init) {
						if (sum == 0) {
							cc.r.center.y = cc.r.center.y - fast_position_step;
						}
						else {
							cc.r.center.y = cc.r.center.y - position_step;
						}
						displayRR(cc);
					}

				}
				else if (k == 24 /*ctrl+x, downscale size*/) {
					//cout << "FN+ Down" << endl;
					if (cc.init) {
						if (sum == 0) {
							height /= fast_scale_factor;
							width /= fast_scale_factor;
							//cout << "New size : " << int(width) << "," << int(height) << endl;
						}
						else {
							height /= scale_factor;
							width /= scale_factor;
							//cout << "New size : " << int(width) << "," << int(height) << endl;
						}

						displayRR(cc);
					}
				}
				else if (k == 2621440 /*down-arrow, shift down*/) {
					//cout << "Down" << endl;
					if (cc.init) {
						if (sum == 0) {
							cc.r.center.y = cc.r.center.y + fast_position_step;
						}
						else {
							cc.r.center.y = cc.r.center.y + position_step;
						}
						displayRR(cc);
					}
				}
				else if (k == 4 /*ctrl+d, rotate right*/) {
					//cout << "FN+ right  << endl;
					if (cc.init) {
						if (sum == 0) {
							orientation -= fast_orientation_step;
						}
						else
							orientation -= orientation_step;
						displayRR(cc);
					}
				}
				else if (k == 2424832 /*right-arrow, shift right*/) {
					//cout << "Right" << endl;
					if (cc.init) {
						if (sum == 0) {
							cc.r.center.x = cc.r.center.x - fast_position_step;
						}
						else
							cc.r.center.x = cc.r.center.x - position_step;

						displayRR(cc);
					}
				}
				else if (k == 1 /*ctrl+a, rotate left*/) {
					//cout << "FN+ left" << endl;
					if (cc.init) {
						if (sum == 0) {
							orientation += fast_orientation_step;
						}
						else
							orientation += orientation_step;

						displayRR(cc);
					}

				}
				else if (k == 2555904 /*left-arrow, shift left*/) {

					//cout << "Left" << endl;
					if (cc.init) {
						if (sum == 0) {
							cc.r.center.x = cc.r.center.x + fast_position_step;
						}
						else
							cc.r.center.x = cc.r.center.x + position_step;

						displayRR(cc);
					}

				}
				else if (k == 262243 /*ctrl+c, terminate*/) {
					// CTRL C
					cout << "Exit" << endl;
					LOG(INFO) << "KeyInterrupt, quit application";
					isBrokenUp = true;
					break;
				}
				else if (k == 27 /*ESC, next image*/) {
					cout << "ESC" << endl;
					digit_number_count = 0;

					numOfImages--;
					LOG(INFO) << "Number of files left: " << numOfImages  << endl;
					if (cc.init) {
						cout << "next rectangle " << endl;
						// next rectangle
						cc.init = false;
						displayRR(cc);
					}
					else  
						break; // next image
				}
				else if (k == 127 || k == 65288 /*erase*/) {

					cout << "Erease" << endl;
					cc.r = RotatedRect(Point2f(0, 0), Size(10, 10), 0);
					displayRR(cc);

				}
				else if (k == 13 || k == 10 /*enter, save*/) { 
					// Enter
					saveRR(cc, image_path, image.factor, classe, &outfile);
				} 
				else {
					//label
					classe = (char)k;
					cout << "Class :" << classe << endl;
					saveRR(cc, image_path, image.factor, classe, &outfile);

					digit_number_count += 1;

					if (digit_number == digit_number_count)
					{
						digit_number_count = 0;
						numOfImages--;
						LOG(INFO) << "Number of files left: " << numOfImages << endl;
						if (cc.init) {
							cout << "next rectangle " << endl;
							// next rectangle
							cc.init = false;
							displayRR(cc);
						}
						else
							break; // next image
					}
				}
			}
		}
	}
	outfile.close();
	return 0;
}
