#include "stdafx.h"
#include <string>
#define NOMINMAX
#include "ExtractRect.h"
#include "Output.h"


int extract_image(Output * output, string  filename, Mat image, \
	RotatedRect r, string label, double ratio, \
	double factor, double factor_width, double factor_height, double offset_x, double offset_y, \
	bool skip_rotation, bool full_image, bool add_borders, int samples, \
	double noise_rotation, double noise_translation, double noise_translation_offset, \
	double noise_zoomin, double noise_zoomout, double resize_width, double gaussian_noise, double pepper_noise, RNG *rng) {

	int nb_extrat = 0;
	int rotation = r.angle, res_w = r.size.width, res_h = r.size.height;
	int len = std::max(image.rows, image.cols);
	cv::Point2f pt(len / 2., len / 2.);

	float rotation_in_radian = ((float)rotation) * 3.14159265 / 180.0; //(rotation-180) % 180 + 180
	int res_x = r.center.x + offset_x * res_w * cos(rotation_in_radian) - offset_y * res_w * sin(rotation_in_radian);
	int res_y = r.center.y + offset_x * res_w * sin(rotation_in_radian) + offset_y * res_w * cos(rotation_in_radian);

	//int res_x = r.center.x + offset_x * res_w * cos(rotation_in_radian) - offset_y * res_h * sin(rotation_in_radian);
	//int res_y = r.center.y + offset_x * res_w * sin(rotation_in_radian) + offset_y * res_h * cos(rotation_in_radian);
	// + Point2f( offset_x * res_w, offset_y * res_h )
	cout << "  Extracting labelled rectangle : [" << label << "," << res_x << "," << res_y << "," << res_w << "," << res_h << "," << rotation << "]" << endl;

	/// COMPUTE ROTATION
	int redress_rotation = 0;
	if (!skip_rotation)
		redress_rotation = rotation;

	int nb_sample = 0;
	while (nb_sample < samples) {
		nb_sample++;

		// noise rotation
		float delta_rotation = 0.0;
		if (noise_rotation != 0.0) {
			delta_rotation = -noise_rotation + 2.0 * static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / noise_rotation));
		}

		// redress image...
		cv::Mat r = cv::getRotationMatrix2D(pt, redress_rotation + delta_rotation, 1.0);
		cv::Mat dst;
		add_salt_pepper_noise(dst, 0.3, 0.3, rng);
		cv::warpAffine(image, dst, r, Size(len, len), INTER_LINEAR, cv::BORDER_CONSTANT);

		// ... and adapt annotation to new referentiel of the new image and factor
		RotatedRect extract_rrect(change_ref(cv::Point2f(res_x, res_y), pt.x, pt.y, redress_rotation + delta_rotation) + pt, Size2f(res_w * factor * factor_width, res_h * factor * factor_height), 0.0);
		RotatedRect origin_rrect(extract_rrect.center, extract_rrect.size, -delta_rotation);

		// noise translation
		float delta_x = 0.0, delta_y = 0.0;
		if (noise_translation != 0.0) {
			float noise_translation_range = (noise_translation - noise_translation_offset);
			delta_x = -noise_translation_range * extract_rrect.size.width + 2.0 * static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (noise_translation_range   * extract_rrect.size.width)));
			delta_x += sgn(delta_x) * noise_translation_offset * extract_rrect.size.width;
			delta_y = -noise_translation_range * extract_rrect.size.height + 2.0 * static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (noise_translation_range * extract_rrect.size.height)));
			delta_y += sgn(delta_y) * noise_translation_offset * extract_rrect.size.height;
			extract_rrect.center = extract_rrect.center + Point2f(delta_x, delta_y);
		}

		// noise scale
		float delta_scale = 1.0;
		if (noise_zoomin != 1.0 || noise_zoomout != 1.0) {
			float scale_range = noise_zoomout - 1 / noise_zoomin;
			delta_scale = 1 / noise_zoomin + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / scale_range));
			extract_rrect.size = Size2f(extract_rrect.size.width * delta_scale, extract_rrect.size.height * delta_scale);
		}

		//extract annotation and adapt to new referentiel
		if (!full_image) {
			dst = extractRotatedRect(dst, extract_rrect);
			//adapt the referentiel
			extract_rrect.center = Point2f(dst.cols / 2.0, dst.rows / 2.0);
			origin_rrect.center = Point2f(dst.cols / 2.0, dst.rows / 2.0) + Point2f(-delta_x, -delta_y);
		}

		//resize by adding border (fit ratio) or by stretching
		int left = 0, right = 0, up = 0, down = 0;
		if (resize_width != 0) {
			if (!add_borders) {
				float w = dst.cols, h = dst.rows;
				resize(dst, dst, Size(resize_width, resize_width * ratio), 0.0, 0.0, INTER_LINEAR);
				float factor_x = resize_width / w;
				float factor_y = resize_width * ratio / h;
				origin_rrect.center.x = origin_rrect.center.x * factor_x;
				origin_rrect.center.y = origin_rrect.center.y * factor_y;
				origin_rrect.size = Size(origin_rrect.size.width * factor_x, origin_rrect.size.height * factor_y);
			}
			else {
				float w = dst.cols;
				dst = resizeContains(dst, resize_width, resize_width*ratio, left, right, up, down, false);
				float factor = dst.cols / w;
				origin_rrect.center.x = origin_rrect.center.x * factor + left;
				origin_rrect.center.y = origin_rrect.center.y * factor + up;
				origin_rrect.size = origin_rrect.size * factor;
			}
		}

		// add pepper noise or gaussian noise
		if (pepper_noise != 0.0)
			add_salt_pepper_noise(dst, pepper_noise, pepper_noise, rng);
		if (gaussian_noise != 0.0)
			add_gaussian_noise(dst, 0, gaussian_noise, rng);

		//write rotated image
		output->write(dst, filename, nb_sample, label, origin_rrect, delta_x, delta_y, delta_rotation, delta_scale, left, right, up, down);
		nb_extrat++;
		// outfile << (std::to_string(item) + ".jpg") << " 1 " << ((int)(p2.x - stoi(result[4])/2)) << " " << ((int)(p2.y - stoi(result[5])/2 )) << " " << stoi(result[4]) << " " << stoi(result[5]) << endl;

		// WorkImage pltimg( output_path);
		// // write image with rotated rect display for debugging
		// displayRotatedRectangle( pltimg.image , RotatedRect( Point2f(rrect.center.x * pltimg.factor, rrect.center.y * pltimg.factor), Size2f(rrect.size.width * pltimg.factor,rrect.size.height * pltimg.factor), rrect.angle ) );
		// output_path = string(argv[2]) + "/_" + std::to_string(i) + ".jpg" ;
		// imwrite( output_path , pltimg.image  );
		// pltimg.release();
		r.release();
		dst.release();
	}
	return nb_extrat;
}


RotatedRect compute_max_bounding_box(std::vector<RotatedRect> &input_rotatedrects) {
	// calcul des coordonnées de la bounding box max
	int x_min = INT_MAX, x_max = 0, y_min = INT_MAX, y_max = 0;
	int height = 0;
	int counts = 0;
	float slope = 0;

	for (int i = 0; i < input_rotatedrects.size(); i++) {

		Rect outputRect = input_rotatedrects[i].boundingRect();


		height += outputRect.br().y - outputRect.tl().y;

		if (outputRect.tl().x < x_min)
			x_min = outputRect.tl().x;
		if (outputRect.br().x > x_max)
			x_max = outputRect.br().x;
		if (outputRect.tl().y < y_min)
			y_min = outputRect.tl().y;
		if (outputRect.br().y > y_max)
			y_max = outputRect.br().y;

		int x1 = getCenterX(outputRect);
		int y1 = getCenterY(outputRect);
		if (i + 1 < input_rotatedrects.size())
			for (int j = i + 1; j < input_rotatedrects.size(); j++) {
				int x2 = input_rotatedrects[j].center.x;
				int y2 = input_rotatedrects[j].center.y;
				slope += ((float)(y2 - y1)) / ((float)(x2 - x1));
				counts++;
			}
	}

	// La max bounding box en Rect
	int width = x_max - x_min;
	Rect rr(x_min, y_min, width, y_max - y_min);
	// float fact_x = factor ;
	// float fact_y = factor;
	// // float fact_x = 1.3;
	// // float fact_y = 1.25;
	// Rect rr_a( x_min - ( fact_x - 1.0 ) * width / 2.0 , y_min - ( fact_y - 1.0 ) * (y_max - y_min) / 2.0 , width * fact_x, (y_max - y_min) * fact_y );

	// La max bounding box en RotatedRect
	int center_x = getCenterX(rr);
	int center_y = getCenterY(rr);
	height = ceil(((float)height) / ((float)input_rotatedrects.size()));
	slope = slope / counts;
	double orientation = atan(slope) * 180 / PI;
	return RotatedRect(Point(center_x, center_y), Size(width, height), orientation);
	// RotatedRect rd_a( Point(center_x, center_y), Size(width * 1.1, height *1.2), orientation);
	// cout << "Rectangle : " << rr << endl;
	// outfile << input[cursor][0] << ",0," << rr_a.tl().x + rr_a.size().width/2.0 << "," << rr_a.tl().y + rr_a.size().height/2.0 << "," << rr_a.size().width << "," << rr_a.size().height << "," << orientation << endl;

}


void compute_merge_by_line(std::vector<RotatedRect> &input_rotatedrects, std::vector<string> &input_labels, std::vector<RotatedRect> &merged_rotatedrects, std::vector<string> &merged_labels) {

	int margin = 5;

	float height = 0.0;
	for (int i = 0; i < input_rotatedrects.size(); i++)
		height = std::max((float)(input_rotatedrects[i].center.y + input_rotatedrects[i].size.height / 2.0), height);

	std::vector<float> counts;
	std::vector<std::vector<RotatedRect> > ordered_rotatedrects;
	std::vector<std::vector<string> > ordered_labels;

	for (int y = 0; y < height - margin; y++) {
		int count = 0;
		std::vector<RotatedRect> line_rotatedrects;
		std::vector<string> line_labels;
		for (int i = 0; i < input_rotatedrects.size(); i++)
			if ((input_rotatedrects[i].center.y >= y) && (input_rotatedrects[i].center.y <= y + margin)) {
				count++;
				line_rotatedrects.push_back(input_rotatedrects[i]);
				line_labels.push_back(input_labels[i]);
			}
		counts.push_back(count);
		ordered_rotatedrects.push_back(line_rotatedrects);
		ordered_labels.push_back(line_labels);
		// cout << "line " << y << endl;
		// for (int i = 0 ; i < line_labels.size(); i ++ )
		//   cout << line_labels[i] ;
		// cout << endl;
	}
	std::vector<int> maxY = Argmax(counts, 100);

	std::vector<int> maxYvalid;

	for (int i = 0; i < maxY.size(); i++) {


		int m = maxY[i]; // index de la ligne
		std::vector<string> l = ordered_labels[m];
		std::vector<RotatedRect> o = ordered_rotatedrects[m];

		if (!l.size())
			break;

		bool next = false;
		for (int j = 0; j < maxYvalid.size(); j++)
			if ((m + margin >= maxYvalid[j]) && (m - margin <= maxYvalid[j])) { next = true; break; }

		if (next)
			continue;

		maxYvalid.push_back(m);
		merged_rotatedrects.push_back(compute_max_bounding_box(o));

		int x_previous = 0;
		string label = "";
		int www = 0;
		for (int k = 0; k < o.size(); k++) {
			int x_min = 1000000000;
			int i_min = 0;
			for (int j = 0; j < o.size(); j++)
				if (o[j].center.x < x_min && o[j].center.x > x_previous) {
					x_min = o[j].center.x;
					i_min = j;
					www = o[j].size.width;
				}
			if ((x_previous != 0) && (x_min - x_previous > 0.9 * www))
				label += " ";
			label += l[i_min];
			x_previous = x_min;
		}
		merged_labels.push_back(label);
	}

}


Point compute_negative(int width, int height, int cols, int rows, std::vector<RotatedRect> input_rotatedrects, float f) {
	int tries = 0;

	int hypothenuse = std::sqrt(cols * cols + rows * rows);
	while (tries < 100) {
		int ss_x1 = rand() % (width - hypothenuse) + cols / 2;
		int ss_y1 = rand() % (height - hypothenuse) + rows / 2;
		bool overlap = false;
		for (int i = 0; i < input_rotatedrects.size(); i++) {
			if (input_rotatedrects[i].size.height * input_rotatedrects[i].size.height * f * f + input_rotatedrects[i].size.width * input_rotatedrects[i].size.width *f * f  > int((ss_x1 - input_rotatedrects[i].center.x * f) * (ss_x1 - input_rotatedrects[i].center.x * f) + (ss_y1 - input_rotatedrects[i].center.y * f) * (ss_y1 - input_rotatedrects[i].center.y * f)))
				overlap = true;
		}
		if (!overlap)
			return Point(ss_x1, ss_y1);

		tries++;
	}
	cout << "Compute negative failed." << endl;
	return Point(-1, -1);
}


