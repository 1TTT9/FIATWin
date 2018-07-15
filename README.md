# FIATWin

This is the project called **Fast Image Data Annotation/Extract Tool (FIAT)**, originated from author [christopher5106](https://github.com/christopher5106) project [FIAT](https://github.com/christopher5106/FastAnnotationTool/blob/master/README.md). I make this as weekend project and mainly work on building and translating them on **Windows** platform.

It provides two main functions: image annotaiton and extraction. Both help deal with image efficiently in preprocessing time before training and analyzing. 

## Annotation
 In annotation, it does simply annotate image by using rectangle (yellow/blue) and provide enlarge/downsize/rotate/ratio adjustments.

## Extraction 
 It realizes not only image annotation by characters or OCR (optial character reader), but also image classification automatically (speicify folder itself, to save lots of manpower). 

## Note
 - I simply made slightly changes and ported to windows inconvience of win-users like me. 
 - KeyEvent is rewritten mostly as it varies in OS and I tried to migrate it from linux to windows.
 - Some functions are failed to complete due to complicated compiling procedure (e.g., caffe in windows).  