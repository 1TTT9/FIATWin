#pragma once
#ifndef THE_FILE_EXTRACTRECT_H
#define THE_FILE_EXTRACTRECT_H

#include "Utils.h"
#include "Output.h"

int extract_image(Output * output, string  filename, Mat image, \
	RotatedRect r, string label, double ratio, \
	double factor, double factor_width, double factor_height, double offset_x, double offset_y, \
	bool skip_rotation, bool full_image, bool add_borders, int samples, \
	double noise_rotation, double noise_translation, double noise_translation_offset, \
	double noise_zoomin, double noise_zoomout, double resize_width, double gaussian_noise, double pepper_noise, RNG *rng);

RotatedRect compute_max_bounding_box(std::vector<RotatedRect> &input_rotatedrects);

void compute_merge_by_line(std::vector<RotatedRect> &input_rotatedrects, std::vector<string> &input_labels, std::vector<RotatedRect> &merged_rotatedrects, std::vector<string> &merged_labels);

Point compute_negative(int width, int height, int cols, int rows, std::vector<RotatedRect> input_rotatedrects, float f);

#endif THE_FILE_EXTRACTRECT_H