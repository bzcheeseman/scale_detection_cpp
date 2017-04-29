//
// Created by Aman LaChapelle on 4/27/17.
//
// Iris
// Copyright (c) 2017 Aman LaChapelle
// Full license at Iris/LICENSE.txt
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


#include "../include/ObjectDetector.hpp"

ObjectDetector::ObjectDetector(std::vector<std::vector<mmod_rect>> train_boxes): options(train_boxes, 60*40) {

  net = net_type(options);

  try{
    deserialize("mmod_network.dat") >> net;
  }
  catch (...){
    std::cerr << "Pretrained Network Not Found" << std::endl;
    trained = false;
  }

}

void ObjectDetector::train(const std::vector<matrix<rgb_pixel>> &training_data,
                           const std::vector<std::vector<mmod_rect>> bounding_boxes) {

  dnn_trainer<net_type> trainer(net);
  trainer.set_learning_rate(0.1);
  trainer.be_verbose();
  trainer.set_synchronization_file("mmod_sync", std::chrono::minutes(2));
  trainer.set_iterations_without_progress_threshold(300);

  std::vector<matrix<rgb_pixel>> mini_batch_samples;
  std::vector<std::vector<mmod_rect>> mini_batch_labels;
  random_cropper cropper;
  cropper.set_max_object_height(0.5);
  cropper.set_background_crops_fraction(0.0);
  cropper.set_max_rotation_degrees(5.0);
  dlib::rand rnd;

  while(trainer.get_learning_rate() >= 1e-4)
  {
    cropper(25, training_data, bounding_boxes, mini_batch_samples, mini_batch_labels);
    int n = mini_batch_samples.size();
    for (int i = 0; i < n; i++){
      disturb_colors(mini_batch_samples[i], rnd);
    }
    try{
      trainer.train_one_step(mini_batch_samples, mini_batch_labels);
    }
    catch(dlib::impossible_labeling_error &e){ // This can happen when the random cropper does something
                                               // strange, just rerun
      std::cout << e.what() << std::endl;
      std::cout << "Terminating Early, Please Rerun" << std::endl;
      return;

    }

  }
  // Wait for training threads to stop
  trainer.get_net();
  std::cout << "Done Training" << std::endl;

  std::cout << "Training Results: " << test_object_detection_function(net, training_data, bounding_boxes) << std::endl;

  // Save the network to disk
  net.clean();
  serialize("mmod_network.dat") << net;

}

void ObjectDetector::inference(std::vector<matrix<rgb_pixel>> &data,
                               const std::string &out_vid_name,
                               const int &codec,
                               const double &fps,
                               const bool &is_color) {

//  deserialize("mmod_network.dat") >> net;

  assert(trained);

  int n = data.size();

  cv::Mat first_frame (toMat(data[0]));
  std::cout << first_frame.size() << std::endl;
  cv::Size frame_size = cv::Size(first_frame.size());  // get this figured out

  cv::VideoWriter vid_out (out_vid_name, -1, fps, frame_size, is_color);
  if (!vid_out.isOpened()){
    std::cerr << "Couldn't open video writer" << std::endl;
    return;
  }

  if (static_cast<bool>(std::ifstream(out_vid_name))){
    std::remove(out_vid_name.c_str());
  }

  cv::Mat output_overlay, output_img, output_frame;
  for (int i = 0; i < n; i++){
    auto img = data[i];
    pyramid_up(img);
    auto output = net(img);

    for (auto &&d : output) {
      std::cout << d << std::endl;
      draw_rectangle(img, d.rect, bgr_pixel(0, 0, 255), 1);
    }

    output_frame = toMat(img);
    std::cout << img.NC << " " << img.NR << std::endl;
    std::cout << output_frame.size() << std::endl;
    vid_out.write(output_frame);
  }
}
