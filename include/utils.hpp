//
// Created by Aman LaChapelle on 4/27/17.
//
// scale_detector_cpp
// Copyright (c) 2017 Aman LaChapelle
// Full license at scale_detector_cpp/LICENSE.txt
//

/*
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */


#ifndef IRIS_UTILS_HPP
#define IRIS_UTILS_HPP

#include <fstream>
#include <sstream>
#include <thread>
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <dlib/serialize.h>
#include <dlib/data_io.h>

// Convert bgr pixels to grey pixels, helper function for dlib
inline dlib::matrix<unsigned char> bgr2grey(const dlib::matrix<dlib::bgr_pixel> &img) {
  dlib::matrix<unsigned char> out (img.nr(), img.nc());

  int rows = img.nr(); int cols = img.nc();
  for (int i = 0; i < rows; i++){
    for (int j = 0; j < cols; j++){
      unsigned char temp;
      dlib::assign_pixel(temp, img(i, j));
      out(i, j) = temp;
    }
  }

  return out;
}

// Convert bgr pixels to rbg pixels, helper function for dlib
inline dlib::matrix<dlib::rgb_pixel> bgr2rgb(const dlib::matrix<dlib::bgr_pixel> &img) {
  dlib::matrix<dlib::rgb_pixel> out (img.nr(), img.nc());

  int rows = img.nr(); int cols = img.nc();
  for (int i = 0; i < rows; i++){
    for (int j = 0; j < cols; j++){
      dlib::rgb_pixel temp;
      dlib::assign_pixel(temp, img(i, j));
      out(i, j) = temp;
    }
  }

  return out;
}

// Unused - reads the object centers from a file
inline void read_img_centers(const std::string &filename, std::vector<dlib::point> &centers){

  std::ifstream file (filename);
  std::string buf;
  long x; long y; dlib::point temp;
  while (std::getline(file, buf)){
    std::stringstream ss(buf);
    ss >> x >> y;
    temp.x() = x;
    temp.y() = y;
    centers.push_back(temp);
  }
}

// Unused - calculates the object centers from the rectangles of their bounding boxes
inline void calc_img_centers(const std::vector<std::vector<dlib::mmod_rect>> &boxes, std::vector<dlib::point> &centers){
  long n = boxes.size();
  for (int i = 0; i < n; i++){
    dlib::rectangle r = boxes[i][0].rect;
    centers.push_back({(long)(r.left()+r.width()/2), (long)(r.top()+r.height()/2)});
  }
}

// Splits a video into individual images for training
inline void video_to_imgs(const std::string &filename,
                          const std::string &output_folder,
                          int &codec, double &fps){
  cv::VideoCapture cap(filename);
  if (!cap.isOpened())
  {
    std::cerr << "Couldn't find video" << std::endl;
    return;
  }

  codec = (int)cap.get(CV_CAP_PROP_FOURCC);
  fps = cap.get(CV_CAP_PROP_FPS);

  std::size_t num_frames = cap.get(CV_CAP_PROP_FRAME_COUNT);
  std::string name (output_folder+"/frame_");
  for (size_t i = 0; i < num_frames; i++){
    cv::Mat temp;
    cap >> temp;
    cv::imwrite(name+std::to_string(i)+".jpg", temp);
  }
}

// Unused - Processes data from the VOT challenge with the idea of using it to train the detector.
inline void process_vot_sequence(const std::string &sequence_name,
                                 std::vector<dlib::matrix<dlib::rgb_pixel>> &imgs,
                                 std::vector<std::vector<dlib::mmod_rect>> &labels,
                                 std::vector<dlib::point> &centers){

  std::ifstream labels_file ("../data/vot2013/" + sequence_name + "/groundtruth.txt");
  std::string buf;
  cv::Mat img;
  std::string frameno;
  float left, top, width, height;
  char comma1, comma2, comma3;
  int i = 0;
  std::vector<dlib::mmod_rect> label_vect;
  while (std::getline(labels_file, buf)){
    std::stringstream ss (buf);
    ss >> left >> comma1 >> top >> comma2 >> width >> comma3 >> height;

    // Add rectangle to mmod vector
    label_vect.push_back(mmod_rect(dlib::rectangle((int)left, (int)top, (int)(left + width), (int)(top + height))));
    labels.push_back(label_vect);
    label_vect.clear();

    // Add center to vector of centers
    centers.push_back({(long)(left+width/2), (long)(top+height/2)});

    // Add image to imgs_vid_3
    std::stringstream imgno;
    imgno << std::setfill('0') << std::setw(8) << i+1;
    img = cv::imread("../data/vot2013/" + sequence_name + "/" + imgno.str() + ".jpg", CV_LOAD_IMAGE_COLOR);
    dlib::cv_image<dlib::bgr_pixel> cvimg(img);
    imgs.push_back(bgr2rgb(dlib::mat(cvimg)));
    i++;
  }

}

// Unused - adds the center point as a bright pixel to the images and returns the modified images
inline void add_center_pt(std::vector<dlib::matrix<dlib::rgb_pixel>> &image,
                          const std::vector<dlib::point> &obj_center) {
  long n = image.size();
  assert(image.size() == obj_center.size());

  dlib::image_window win;
  for (int i = 0; i < n; i++){
    image[i] = dlib::trans(image[i]);
    dlib::rgb_pixel &pixel = image[i](obj_center[i].x(), obj_center[i].y());
    dlib::assign_pixel_intensity(pixel, 255);
    image[i] = dlib::trans(image[i]);
  }

}

// Loads data from the training datasets, tried with VOT data but found that causes problems so no longer.
// Discovered a bug that means that only training data from video3 get loaded.
inline void load_train_data(const std::vector<std::string> &vot_sequences,
                            std::vector<dlib::matrix<dlib::rgb_pixel>> &imgs,
                            std::vector<std::vector<dlib::mmod_rect>> &labels,
                            std::vector<dlib::point> &centers){
  // Get training data from videos
  dlib::load_image_dataset(imgs, labels, "../data/video2.xml");  // This one doesn't get loaded it looks like :(
  calc_img_centers(labels, centers);
  dlib::load_image_dataset(imgs, labels, "../data/video3.xml");
  calc_img_centers(labels, centers);
}

// Loads testing data, takes output of video_to_imgs and reads it into a vector of samples.
inline void load_test_data(const std::string &folder,
                           const long &num_imgs,
                           std::vector<dlib::matrix<dlib::rgb_pixel>> &test_samples){

  for (long i = 0; i < num_imgs; i++){
    cv::Mat img;
    img = cv::imread(folder+"/frame_"+std::to_string(i)+".jpg");
    dlib::cv_image<dlib::bgr_pixel> cvimg(img);
    test_samples.push_back(bgr2rgb(dlib::mat(cvimg)));
  }
}

#endif //IRIS_UTILS_HPP
