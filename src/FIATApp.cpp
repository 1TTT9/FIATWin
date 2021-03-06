// FIATApp.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include  "Utils.h"

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "AnnotateRect.h"

#include "ExtractRect.h"


using std::cout;
using std::endl;

#pragma region Parameters General
DEFINE_int32(tooloption, 1, "Choose to use either 1:Annotation or 2:ExtractRect, default is 1:Annotation");
// number of digit to capture
DEFINE_int32(digit, 1, "Number of digit you wanna caputre, default is 1");
// Capture window dimensions
DEFINE_double(ratio, 1.0, "The ratio of capture window height/width");
#pragma endregion Parameters General

#pragma region Parameters Annotation
DEFINE_string(init, "", "Initialize with rectangles");

DEFINE_bool(cross, false, "Display of rectangle other than current");
DEFINE_string(export, "", "Export drawn rectangles on images");
#pragma endregion Parameters Annotation

#pragma region Parameter ExtractArea
// input
DEFINE_string(input_class_filter, "", "Filter entries on the specified class");
DEFINE_int32(limit, 0, "The number of input annotations to consider");

//offsets
DEFINE_double(offset_x, 0.0, "Add an offset on the first axis of the rectangle, in percentage of the width of the rectangle. If zero, no offset");
DEFINE_double(offset_y, 0.0, "Add an offset on the second axis of the rectangle, in percentage of the width of the height. If zero, no offset");

DEFINE_double(factor, 1.0, "The scale factor to apply on annotation rectangle width and height");
DEFINE_double(factor_width, 1.0, "The scale factor to apply on annotation rectangle width");
DEFINE_double(factor_height, 1.0, "The scale factor to apply on annotation rectangle height");

// operations
// DEFINE_bool(redress, true, "Image is rotated to be parallel to the annotation window [Default : true]");
DEFINE_bool(merge, false, "If multiple rectangle per images, merge them");
DEFINE_bool(merge_line, false, "If multiple rectangle per images, merge rectangles that are roughly on the same line.");
DEFINE_bool(correct_ratio, false, "Correct the ratio of the annotated rectangles by augmenting its smallest dimension");
DEFINE_bool(add_borders, false, "Add borders to the window to fit the ratio");
DEFINE_bool(skip_rotation, false, "Skip rotation angle");
DEFINE_bool(full_image, false, "Will not extract the annotation");

// noise
DEFINE_double(noise_rotation, 0.0, "Add a rotation noise. If zero, no noise");
DEFINE_double(noise_translation, 0.0, "Add a translation noise. In %age of the dimensions. If zero, no noise");
DEFINE_double(noise_translation_offset, 0.0, "Defines an offset in the translation noise. In %age of the dimensions. If zero, no offset");
DEFINE_double(noise_zoomin, 1.0, "Add a noise in the scale factor. If 1.0, no zoomin noise");
DEFINE_double(noise_zoomout, 1.0, "Add a noise in the scale factor. If 1.0, no zoomout noise");
DEFINE_int32(samples, 1, "The number of noised samples to extract");
DEFINE_double(pepper_noise, 0.0, "Add pepper noise");
DEFINE_double(gaussian_noise, 0.0, "Add gaussian noise");

// output
DEFINE_double(resize_width, 0.0, "Resize width of capture window. If zero, no resize");
DEFINE_bool(gray, false, "Extract as a gray image");
DEFINE_string(backend, "directory", "The output format for storing the result. Possible values are : directory, lmdb, tesseract, opencv");
DEFINE_string(output_class, "", "Override the class by the specified class");
DEFINE_bool(output_by_label, true, "Output different labels in different output directories. For backend=directory only.");
DEFINE_bool(append, false, "Append results to an existing directory. For backend=directory only.");

// negatives
DEFINE_double(neg_width, 0.2, "The width of negative samples to extract, in pourcentage to the largest image dimension (width or height)");
DEFINE_int32(neg_per_pos, 0, "The number of negative samples per positives to extract");
#pragma endregion Parameter ExtractArea


void InitLogger(char** argv)
{
	google::InitGoogleLogging(argv[0]);
	google::SetLogDestination(google::GLOG_INFO, "D:\\LocalLog\\INFO_");
	google::SetStderrLogging(google::GLOG_INFO);
	google::SetLogFilenameExtension("log_");
	FLAGS_colorlogtostderr = true;  // Set log color
	FLAGS_logbufsecs = 0;  // Set log output speed(s)
	FLAGS_max_log_size = 1024;  // Set max log file size
	FLAGS_stop_logging_if_full_disk = true;  // If disk is full
}

int main(int argc, char** argv)
{
#ifndef GFLAGS_GFLAGS_H_
	namespace gflags = google;
#endif
	gflags::SetUsageMessage("This script to annotate rectangles to\n"
		"Usage:\n"
		"    [1:Annotation] input_dir output_file.csv [FLAGS]\n"
		"    [2:ExtractRect] input.csv output_dir [FLAGS]\n"
		"\n");
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	InitLogger(argv);
	LOG(INFO) << "argc: " << argc;
	LOG(INFO) << "argv[0]: " << argv[0];

	if (argc != 3) {
		gflags::ShowUsageWithFlagsRestrict(argv[0], "FIATApp");
		LOG(WARNING) << "insufficent paraeters, application will close";
		google::ShutdownGoogleLogging();
		return -1;
	}

	const int& toolOption = FLAGS_tooloption;
	LOG(INFO) << "Tool option: " << toolOption << endl;

	const int& numberdigit = FLAGS_digit;
	LOG(INFO) << "Digital number: " << numberdigit << endl;

#pragma region Annotation
	const string& boxes = FLAGS_init;
	const double& ratio = FLAGS_ratio;
	// // const int& width = FLAGS_width;
	// // const int& height = width * ratio ;
	// const int& factor = FLAGS_factor;
	// const int& max = FLAGS_max;
	// const int& min = FLAGS_min;
	// const int& scales = FLAGS_scales;
	// const bool& rotated = FLAGS_rotated;
	// const bool& correct = FLAGS_correct;
	const bool& cross = FLAGS_cross;
	const string& export_dir = FLAGS_export;
	LOG(INFO) << "Annotation: " << endl << "=====================================";
	LOG(INFO) << "init: " << boxes;
	LOG(INFO) << "ratio: " << ratio;
#pragma endregion Annotation

#pragma region ExtractRect
	//const double& ratio = FLAGS_ratio;
	double& factor = FLAGS_factor;
	double& factor_width = FLAGS_factor_width;
	double& factor_height = FLAGS_factor_height;
	double& offset_x = FLAGS_offset_x;
	double& offset_y = FLAGS_offset_y;

	const bool& merge = FLAGS_merge;
	const bool& merge_line = FLAGS_merge_line;
	const bool& correct_ratio = FLAGS_correct_ratio;
	const double& pepper_noise = FLAGS_pepper_noise;
	const double& gaussian_noise = FLAGS_gaussian_noise;
	const double& noise_rotation = FLAGS_noise_rotation;
	const double& noise_translation = FLAGS_noise_translation;
	const double& noise_translation_offset = FLAGS_noise_translation_offset;
	const double& noise_zoomin = FLAGS_noise_zoomin;
	const double& noise_zoomout = FLAGS_noise_zoomout;
	double& resize_width = FLAGS_resize_width;
	const int& samples = FLAGS_samples;
	const double& neg_width = FLAGS_neg_width;
	const int& neg_per_pos = FLAGS_neg_per_pos;
	const bool& add_borders = FLAGS_add_borders;
	bool& gray = FLAGS_gray;
	const string& db_backend = FLAGS_backend;
	const string& input_class_filter = FLAGS_input_class_filter;
	const int& limit = FLAGS_limit;

	const string& output_class = FLAGS_output_class;
	const bool& output_by_label = FLAGS_output_by_label;
	const bool& append = FLAGS_append;
	const bool& skip_rotation = FLAGS_skip_rotation;
	bool& full_image = FLAGS_full_image;
#pragma endregion ExtractRect

	string base_path = getbase(argv[1]);
	LOG(INFO) << "Base path: " << base_path;

	if (toolOption == 1 /* Annotation */)
	{
		if (export_dir != "")
#ifdef _WINDOWS
			CHECK_EQ(_mkdir(export_dir.c_str()), 0)
#else
			CHECK_EQ(_mkdir(export_dir.c_str(), 0744), 0)
#endif
			<< "mkdir " << export_dir << " failed";
		annotate(argv[1], argv[2], ratio, boxes, cross, export_dir, numberdigit);
	}
	else if (toolOption == 2 /* ExtractRect */)
	{
		LOG(INFO) << "output dir: " << argv[2] << endl;

		Output * output;
		if (db_backend == "directory") {
			output = new Directory(argv[2], append, output_by_label);
		}
		else if (db_backend == "opencv") {
			full_image = true;
			gray = true;
			output = new OpnCV(argv[2]);
#ifdef CAFFE
		}
		else if (db_backend == "lmdb") {
			output = new LMDB(argv[2]);
#endif
		}
		else if (db_backend == "tesseract") {
			output = new Tessract(argv[2]);
		}
		else
			LOG(FATAL) << "Unknown db backend " << db_backend;

		string base_path = getbase(argv[1]);
		cout << "Base path: " << base_path << endl;
		cout << "Capture window ratio : " << ratio << endl;
		cout << "Capture window scale factor : " << factor << endl;
		if (gray)
			cout << "Extract in gray (1 channel)" << endl;
		if (skip_rotation)
			cout << "Rotation information in annotations will be skipped." << endl;
		if (add_borders)
			cout << "Add borders to fit ratio." << endl;
		cout << "Noise rotation : [-" << noise_rotation << "," << noise_rotation << "]" << endl;
		cout << "Noise translation : [-" << noise_translation << "," << noise_translation << "]" << endl;
		cout << "Number of samples : " << samples << endl;


		std::vector<std::vector<std::string> > input;
		readCSV(argv[1], input);

		// get rectangles for each unique input_path
		std::vector<std::string> input_path;
		std::vector<std::vector<string> > input_labels;
		std::vector<std::vector<RotatedRect> > input_rotatedrects;
		bool consider_rotation = true;
		group_by_image(input, consider_rotation, !merge && correct_ratio, ratio, input_path, input_labels, input_rotatedrects);

		RNG rng;
		int nb_inputs = 0;

		for (int cursor = 0; cursor < input_path.size(); cursor++) {

			if (limit != 0 && nb_inputs > limit) break;

			//load image
			string image_path = input_path[cursor];
			//if (image_path.at(0) != '.' && image_path.at(0) != '/')
			//	image_path = base_path + image_path;

			Mat image = imread(image_path, CV_LOAD_IMAGE_COLOR);
			if (!image.data)                              // Check for invalid input
			{
				//std::cout << "Could not open or find the image " << base_path + input_path[cursor] << std::endl;
				cout << "Could not open or find the image " << input_path[cursor] << std::endl;
			}
			else {
				cout << "Item " << ": " << input_path[cursor] << endl;

				float f = 1.0;
				int max_dim = max(image.rows, image.cols);
				if (max_dim>10000) {
					cout << "Image too big, resizing." << endl;
					f = 5000.0 / ((float)max_dim);
					resize(image, image, Size(), f, f, INTER_AREA);
				}

				if (gray)
					cvtColor(image, image, CV_BGR2GRAY);

				string filename = input_path[cursor];
				filename.erase(filename.end() - 4, filename.end());
				filename = filename.substr(filename.find_last_of("/") + 1);

				// compute the positives
				int nb_positives = 0;
				if (!merge && !merge_line)
					for (int i = 0; i < input_rotatedrects[cursor].size(); i++) {
						string label = input_labels[cursor][i];
						if (input_class_filter != "" && label != input_class_filter) continue;
						if (output_class != "") label = output_class;

						RotatedRect r = input_rotatedrects[cursor][i];
						r.size.height = r.size.height * f;
						r.size.width = r.size.width * f;
						r.center.x = r.center.x * f;
						r.center.y = r.center.y * f;

						nb_positives += extract_image(output, filename + "_" + std::to_string(i), image, \
							r, label, ratio, \
							factor, factor_width, factor_height, offset_x, offset_y, \
							skip_rotation, full_image, add_borders, samples, \
							noise_rotation, noise_translation, noise_translation_offset, \
							noise_zoomin, noise_zoomout, resize_width, gaussian_noise, pepper_noise, &rng);
					}
				else {
					std::vector<string> merged_labels;
					std::vector<RotatedRect> merged_rotatedrects;
					if (merge_line) {
						cout << "Merge lines :" << endl;
						compute_merge_by_line(input_rotatedrects[cursor], input_labels[cursor], merged_rotatedrects, merged_labels);
						for (int i = 0; i < merged_labels.size(); i++)
							cout << "   " << merged_labels[i] << endl;
					}
					else {
						merged_rotatedrects.push_back(compute_max_bounding_box(input_rotatedrects[cursor]));
						merged_labels.push_back("0");
					}
					for (int i = 0; i < merged_rotatedrects.size(); i++) {
						RotatedRect r = merged_rotatedrects[i];
						r.size.height = r.size.height * f;
						r.size.width = r.size.width * f;
						r.center.x = r.center.x * f;
						r.center.y = r.center.y * f;
						nb_positives += extract_image(output, filename + "_" + std::to_string(i), image, \
							r, merged_labels[i], ratio, \
							factor, factor_width, factor_height, offset_x, offset_y, \
							skip_rotation, full_image, add_borders, samples, \
							noise_rotation, noise_translation, noise_translation_offset, \
							noise_zoomin, noise_zoomout, resize_width, gaussian_noise, pepper_noise, &rng);
					}


				}
				nb_inputs += nb_positives;

				// compute the negatives
				int neg = 0;
				while (neg < neg_per_pos * nb_positives) {
					int max_dim = std::max(image.cols, image.rows);
					// cout << "Width of negative extract : " << max_dim * neg_width << endl;
					Point neg_po = compute_negative(image.cols, image.rows, max_dim * neg_width, max_dim * neg_width * ratio, input_rotatedrects[cursor], f);
					if (neg_po.x != -1) {
						neg += extract_image(output, filename + "_" + std::to_string(neg), image, \
							RotatedRect(neg_po, Size2f(max_dim *neg_width, max_dim * neg_width * ratio), 0.0), "", ratio, \
							1.0, 1.0, 1.0, 0.0, 0.0, \
							skip_rotation, full_image, add_borders, 1, \
							noise_rotation, 0.0, 0.0, \
							1.0, 1.0, resize_width, gaussian_noise, pepper_noise, &rng);
					}
				}

				image.release();
			}
		}
		output->close();
	}
	google::ShutdownGoogleLogging();
	return 0;
}