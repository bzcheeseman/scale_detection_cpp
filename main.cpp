#include <iostream>

#include "include/ObjectDetector.hpp"
#include "include/utils.hpp"

int main() {

  std::vector<matrix<rgb_pixel>> imgs, test_imgs;
  std::vector<std::vector<mmod_rect>> labels;
  std::vector<point> centers;

  bool reload = false;

  try{
    if (reload){
      std::remove("../data/train_data.dat");
      std::remove("../data/train_labels.dat");
      std::remove("../data/train_centers.dat");
    }
    deserialize("../data/train_data.dat") >> imgs;
    deserialize("../data/train_labels.dat") >> labels;
    deserialize("../data/train_centers.dat") >> centers;
  }
  catch (...){
    load_train_data({"jump"}, imgs, labels, centers);  // load up training data - add "car" sequence? put back VOT data?
    serialize("../data/train_data.dat") << imgs;
    serialize("../data/train_labels.dat") << labels;
    serialize("../data/train_centers.dat") << centers;

  }
//
//  std::cout << "Starting Training" << std::endl;

  ObjectDetector detector(labels);
//  detector.train(imgs, labels);

  int codec;
  double fps;

  std::cout << "Loading Test Data" << std::endl;

  bool reload_test = false;

  try{
    if (reload_test){
      std::remove("../data/test_data.dat");
      std::remove("../data/test_codec.dat");
      std::remove("../data/test_fps.dat");
    }
    deserialize("../data/test_data.dat") >> test_imgs;
    deserialize("../data/test_codec.dat") >> codec;
    deserialize("../data/test_fps.dat") >> fps;
  }
  catch(...){
    video_to_imgs("../data/test/video3.mp4", "../data/imgs_vid_3_test", codec, fps);
    load_test_data("../data/imgs_vid_3_test", 596, test_imgs);  // load up test data - this takes a while
    serialize("../data/test_data.dat") << test_imgs;
    serialize("../data/test_codec.dat") << codec;
    serialize("../data/test_fps.dat") << fps;
  }

  std::cout << "Starting Inference" << std::endl;

  detector.inference(test_imgs, "../data/video3", codec, fps, true);

  return 0;
}