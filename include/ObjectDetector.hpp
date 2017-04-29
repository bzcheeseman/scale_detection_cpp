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


#ifndef IRIS_OBJECTDETECTOR_HPP
#define IRIS_OBJECTDETECTOR_HPP

#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>

using namespace dlib;

class ObjectDetector {

  // Define conv blocks here, first doesn't downsample, second does
  template<long num_filters, typename SUBNET> using conv5 = con<num_filters, 5, 5, 1, 1, SUBNET>;
  template<long num_filters, typename SUBNET> using conv5down = con<num_filters, 5, 5, 2, 2, SUBNET>;

  // Define input stem, takes in input layer and downsamples 8x
  template<typename SUBNET>
  using stem = relu<bn_con<conv5down<32, relu<bn_con<conv5down<32, relu<bn_con<conv5down<16, SUBNET>>>>>>>>>;

  // Define net block with non-downsampling conv blocks
  template<typename SUBNET> using block = relu<bn_con<conv5<45, SUBNET>>>;

  /*
   * Define network, 8x downsample followed by 3 net blocks followed by 7x7 conv
   * loss_mmod is the implementation of the algorithm by Davis King - Min. Margin Object Detection
   * The input layer is
   */
  using net_type =
    loss_mmod<con<1, 7, 7, 1, 1, block<block<block<stem<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;


  mmod_options options; // MMOD options
  net_type net; // Network
  bool trained = true;

public:
  /**
   * @brief Constructor
   * @param train_boxes The bounding boxes for the training samples - allows us to set the MMOD options correctly.
   */
  ObjectDetector(std::vector<std::vector<mmod_rect>> train_boxes);

  /**
   * @brief Trains the network
   * @param training_data A vector of the training images
   * @param bounding_boxes A vector of the bounding boxes in each image
   */
  void train(const std::vector<matrix<rgb_pixel>> &training_data,
             const std::vector<std::vector<mmod_rect>> bounding_boxes);

  /**
   * @brief Runs inference on a vector of images and (hopefully) outputs a video
   * @param data Vector of images
   * @param out_vid_name Name of output video
   * @param codec Video codec - must be set by input (since we're just placing an overlay on the images),
   *              comes from video_to_imgs. Currently just using -1 for system default.
   * @param fps Video fps - also set by input, comes from video_to_imgs
   * @param is_color If the video is supposed to have color or not. In this case, it is (the bounding boxes are red)
   */
  void inference(std::vector<matrix<rgb_pixel>> &data,
                 const std::string &out_vid_name,
                 const int &codec,
                 const double &fps,
                 const bool &is_color);

};


#endif //IRIS_OBJECTDETECTOR_HPP
