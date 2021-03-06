# FIATWin

This is a mimic project  **Fast Image Data Annotation/Extract Tool (FIAT)**, originated from author [christopher5106](https://github.com/christopher5106) project [FIAT](https://github.com/christopher5106/FastAnnotationTool/blob/master/README.md). I work this as weekend project and mainly focus on building and translating them on **Windows** platform.

It provides two main functions: image annotaiton and extraction. Both help deal with image efficiently in preprocessing time before training and analyzing. 

## Prerequisites
	- Opencv>=3.1.0
	- glog>=3.6.0
	- gflags>=2.2.1
	- protobuf>=3.6
	- dirent

## Usage
	```	
		--tooloption=[1:Annotation] input_dir output_file.csv [FLAGS]
		--tooloption=[2:ExtractRect] input.csv output_dir [FLAGS]
	```

## Flags
	General
	```
		--tooloption 1, Choose to use either 1:Annotation or 2:ExtractRect, default is 1:Annotation
		--digit 1, Number of digits you wanna caputre, default is 1
		--ratio 1.0, The ratio of capture window height/width
	```
	Annotation
	```
		--init "", Initialize with rectangles
		--cross false, Display of rectangle other than current
		--export "", Export drawn rectangles on images
	```
	Extraction
	```
		--input_class_filter "", Filter entries on the specified class
		--limit 0, The number of input annotations to consider
		--offset_x 0, Add an offset on the first axis of the rectangle, in percentage of the width of the rectangle. If zero, no offset
		--offset_y 0, Add an offset on the second axis of the rectangle, in percentage of the width of the height. If zero, no offset
		--factor 1.0, The scale factor to apply on annotation rectangle width and height
		--factor_width 0, The scale factor to apply on annotation rectangle width
		--factor_height 0, The scale factor to apply on annotation rectangle height
		--merge false, If multiple rectangle per images, merge them
		--merge_line false, If multiple rectangle per images, merge rectangles that are roughly on the same line.
		--correct_ratio false, Correct the ratio of the annotated rectangles by augmenting its smallest dimension
		--add_borders false, Add borders to the window to fit the ratio
		--skip_rotation false, Skip rotation angle
		--full_image false, Will not extract the annotation
		--noise_rotation 0.0, Add a rotation noise. If zero, no noise
		--noise_translation 0.0, Add a translation noise. In %age of the dimensions. If zero, no noise
		--noise_translation_offset 0.0, Defines an offset in the translation noise. In %age of the dimensions. If zero, no offset
		--noise_zoomin 1.0, Add a noise in the scale factor. If 1.0, no zoomin noise
		--noise_zoomout 1.0, Add a noise in the scale factor. If 1.0, no zoomout noise
		--samples 1, "The number of noised samples to extract
		--pepper_noise 0.0, Add pepper noise
		--gaussian_noise 0.0, Add gaussian noise
		--resize_width 0.0, "Resize width of capture window. If zero, no resize
		--gray, false "Extract as a gray image
		--backend "directory", The output format for storing the result. Possible values are : directory, lmdb, tesseract, opencv
		--output_class "", Override the class by the specified class
		--output_by_label true, Output different labels in different output directories. For backend=directory only
		--append false, Append results to an existing directory. For backend=directory only
		--neg_width 0.2, The width of negative samples to extract, in pourcentage to the largest image dimension (width or height)
		--neg_per_pos 0, The number of negative samples per positives to extract
	```




## Annotation
 In annotation, it does simply annotate image by using rectangle (yellow/blue) and provide enlarge/downsize/rotate/ratio adjustments.

## Extraction 
 It realizes not only image annotation by characters or OCR (optial character reader), but also image classification automatically (speicify folder itself, to save lots of manpower). 

## Note
 - I made slightly changes and ported them to windows in convience of win-users like me. 
 - KeyEvent is rewritten mostly as it varies in OS and I tried to migrate it from linux to windows.
 - Some functions are failed to complete due to complicated compiling procedure (e.g., caffe in windows).  