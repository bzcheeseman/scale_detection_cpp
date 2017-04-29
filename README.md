# Scale Detection

I could have used any number of approaches using the Dlib framework. Dlib offers detectors like correlation trackers, 
which are given the bounding box at the beginning and then attempt to track the object through a video, as well as full 
object detectors that are based on FHOG detectors. However, as I'm interviewing for a Deep Learning position, I chose 
to use a CNN to scan the image and find the object that way. This also means that I didn't need to use the center of
the object information.

Unfortunately for me, this approach takes a long time and a lot of data to train - I had neither. I made a number of
changes to compensate for this: I took video2 and turned it into training data. I took video3 and randomly selected
frames and labeled them manually for use in training. I did not do anything with video1, though in retrospect I should
have done the same as I did in video3. In order to speed up training, I used an algorithm developed by the dlib creator
Davis King called MMOD, or Min. Margin Object Detection. Davis was able to create a face detector with only 4 images
in just a few hours - this is perfect for both the size of data I have available and the amount of time I have.

The 'few hours' time was done with a Titan X GPU, however. Since the latest Xcode update, my CUDA acceleration hasn't
been working in Dlib at all. I trained the model for about 14 hours to get a workable detector. The model was able to
accurately detect the helicopter (when it was big enough) for all the training images, which we would expect. The 
model's bounding boxes did not exactly match the boxes that I had labeled with, but they still completely contained
the helicopter.

When I started running tests on the trained model, I found that the model wouldn't detect anything for any videos other
than video3. There had been an unexpected result in my code where the load_image_dataset function from Dlib cleared
the vector of images before adding to it, so the network only got trained on the images from video3. This is unfortunate
because it means we can't use this on the other videos (at least, I wouldn't expect it to work because it's never
seen an airplane or a quadcopter), but we can still test it on the entirety of video3 and see if it has learned to 
generalize from the ~20 frames it trained on.

The results are promising - I used an initial guess for size that was too large (60x40 pixels) to make detections 
at the very beginning of the video, when the helicopter is very far away. This could be remedied by shrinking down the
size guess and retraining.

The shortcomings of my approach are that it's slower than the other approaches, especially when you don't have
any hardware acceleration, and training takes much longer than other approaches. On the other hand, given enough time
and data, the CNN approach is much more accurate and can in fact perform under much worse conditions than the other
approaches (as shown by the VOT challenges and other deep learning image detection challenges).
