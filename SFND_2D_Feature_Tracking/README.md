## Camera Based 2D Feature Tracking

### MP.1 Data Buffer Optimization
<font size="3">
Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf.

---
If dataBuffer is changed from vector to a size-fixed ring buffer data structure, the following functions and operators need to be overloaded to track the starting and ending location of the buffer:


* **dataBuffer.end():** For std::vector type buffer, function **end()** always keeps increasing and points to the address imediately after the last element when a new element is added to the buffer. But for a fixed-size ring buffer, the address returned by **end()** function need to be wrapped back to the starting point of buffer and overwrite its content when the buffer is full and a new elements is added;

* **dataBuffer.push_back(frame):** For std::vector container, push_back() function always adds a new element to a location after the last element. While for ring buffer, the new element is added and overwrite the oldest element when buffer is full;

* **(dataBuffer.end() - 1)->cameraImg:** Subtraction “-” operator needs to be overloaded to handle the starting (ending) point when  subtraction result less than 0 or the buffer is full. Ring buffer also need to overload ->” operator. 

In this design, a ring buffer is implemented based on guidance from the following link:
https://users.cs.northwestern.edu/~riesbeck/programming/c++/stl-iterator-define.html#TOC11 to meet all above features
</font>

```python
//Foward Declaration of iterator
template <class T> class RingBufferIterator;
//Ring Buffer
template<class T>
class RingBuffer
{
public:
  friend class RingBufferIterator<T>;
  typedef RingBufferIterator<T> iterator;

  explicit RingBuffer(size_t N)
    :ring_buffer_(nullptr)
    ,buffer_size_(0)
    ,capacity_(N)
    ,begin_(0)
    {  ring_buffer_ = new T[N];
       capacity_ = N;
     };

  size_t size(){return buffer_size_;}
  iterator begin(){return iterator(*this, 0);}
  iterator end(){return iterator(*this, buffer_size_);}

  void push_back(const T& item)
  {
    *(end()) = item;
    if(buffer_size_ < capacity_) ++buffer_size_;
    else begin_ = (begin_ + 1) % capacity_;
  }

private:
  T* ring_buffer_;
  size_t buffer_size_;
  size_t capacity_;
  size_t begin_;
};

//Iterator declaration
template<class T>
class RingBufferIterator{
  private:
    RingBuffer<T>& rb_;
    size_t offset_;
  public:
    RingBufferIterator(RingBuffer<T>& rb, size_t offset)
    :rb_(rb),
    offset_ (offset)
    {
    }

    T& operator*()
    { return rb_.ring_buffer_[(rb_.begin_ + offset_)%rb_.capacity_]; }

    T *operator->()
    {return &(operator*());}

    bool operator==(const RingBufferIterator& it)
    {
      return &(this->rb_) == &(it.rb_) && (this->offset_ == it.offset_);
    }

    bool operator!=(const RingBufferIterator& it)
    {
      return not (*this == it);
    }

    RingBufferIterator& operator+ ( ptrdiff_t delta )
    {
      offset_ += delta;
      return * this;
    }

    RingBufferIterator& operator++ ()
    {
      offset_ += 1;
      return * this;
    }

    RingBufferIterator& operator- ( ptrdiff_t delta )
    {
      offset_ -= delta;
      return * this;
    }
    RingBufferIterator& operator-- ()
    {
      offset_ -= 1;
      return * this;
    }
};

```

## MP.2 Keypoint Detection
<font size="3">
Implement detectors HARRIS, FAST, BRISK, ORB, AKAZE, and SIFT and make them selectable by setting a string accordingly.</font>

---
### HARRIS:
```python
double detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
  // Detector parameters
  int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
  int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
  int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
  double k = 0.04;       // Harris parameter (see equation for details)
  // Detect Harris corners and normalize output
  cv::Mat dst, dst_norm, dst_norm_scaled;
  dst = cv::Mat::zeros(img.size(), CV_32FC1);
  double t = (double)cv::getTickCount();
  cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
  cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
  cv::convertScaleAbs(dst_norm, dst_norm_scaled);
  // Look for prominent corners and instantiate keypoints
  //vector<cv::KeyPoint> keypoints;
  double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression
  for (size_t j = 0; j < dst_norm.rows; j++)
  {
      for (size_t i = 0; i < dst_norm.cols; i++)
      {
          int response = (int)dst_norm.at<float>(j, i);
          if (response > minResponse)
          { // only store points above a threshold

              cv::KeyPoint newKeyPoint;
              newKeyPoint.pt = cv::Point2f(i, j);
              newKeyPoint.size = 2 * apertureSize;
              newKeyPoint.response = response;

              // perform non-maximum suppression (NMS) in local neighbourhood around new key point
              bool bOverlap = false;
              for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
              {
                  double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                  if (kptOverlap > maxOverlap)
                  {
                      bOverlap = true;
                      if (newKeyPoint.response > (*it).response)
                      {                      // if overlap is >t AND response is higher for new kpt
                          *it = newKeyPoint; // replace old key point with new one
                          break;             // quit loop over keypoints
                      }
                  }
              }
              if (!bOverlap)
              {                                     // only add new key point if no overlap has been found in previous NMS
                  keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
              }
          }
      } // eof loop over cols

  }     // eof loop over rows
  t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

  return 1000 * t / 1.0;
}
```
---
### FAST:
```python
double detKeypointsFAST(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
  int threshold = 30;       // difference between intensity of the central pixel and pixels of a circle around this pixel
  bool bNMS = true;
  cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16; // TYPE_9_16, TYPE_7_12, TYPE_5_8
  cv::Ptr<cv::FeatureDetector> detector = cv::FastFeatureDetector::create(threshold, bNMS, type);
  double t = (double)cv::getTickCount();
  detector->detect(img, keypoints);
  t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
  return 1000 * t / 1.0;
}
```
---
### BRISK:
```python
double detKeypointsBRISK(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
  int Thresh = 100;
  int Octave = 4;
  float PatternScales = 1.0f;
  //cv::Ptr<cv::BRISK> detector = cv::BRISK::create(Thresh, Octave, PatternScales);
  cv::Ptr<cv::BRISK> detector = cv::BRISK::create();
  double t = (double)cv::getTickCount();
  detector->detect(img,keypoints);
  t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
  return 1000 * t / 1.0;
}
```
---
### ORB:
```python
double detKeypointsORB(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
  int nFeatures = 100;
  int nLevels = 8;
  float scaleFactor = 1.2f;
  //cv::Ptr<cv::ORB> detector = cv::ORB::create(nFeatures, scaleFactor, nLevels);
  cv::Ptr<cv::ORB> detector = cv::ORB::create();
  double t = (double)cv::getTickCount();
  detector->detect(img,keypoints);
  t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
  return 1000 * t / 1.0;
}
``` 

---
### AKAZE:
```python
double detKeypointsAKAZE(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{

  cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create();
  double t = (double)cv::getTickCount();
  detector->detect(img,keypoints);
  t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
  return 1000 * t / 1.0;
}

```
---
### SIFT:
```python
double detKeypointsSIFT(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
  //int numFeatures = 400;
  cv::Ptr<cv::xfeatures2d::SIFT> detector = cv::xfeatures2d::SIFT::create();
  double t = (double)cv::getTickCount();
  detector->detect(img,keypoints);
  t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
  return 1000 * t / 1.0;
}

```

## MP.3 Keypoint Removal
<font size="3">
Remove all keypoints outside of a pre-defined rectangle and only use the keypoints within the rectangle for further processing.

The following code keeps the keypoints from the previous vehcle only:</font>

```python

void focusOnVehicle(vector<cv::KeyPoint>& keypoints, bool bFocusOnVehicle)
{
  cv::Rect vehicleRect(535, 180, 180, 150);
  if (bFocusOnVehicle)
  {
      vector<cv::KeyPoint>::iterator it = keypoints.begin();
      while(it != keypoints.end())
      {
        if(!vehicleRect.contains((*it).pt))
        {
          it = keypoints.erase(it);
        }else
        {
          it++;
        }
      }
  }
}
```


## MP.4 Keypoint Descriptors
<font size="3">
Implement descriptors BRIEF, ORB, FREAK, AKAZE and SIFT and make them selectable by setting a string accordingly.
</font>

```python
double descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{   //Descriptors: BRIEF, ORB, FREAK, AKAZE, SIFT
    // select appropriate descriptor

    cv::Ptr<cv::DescriptorExtractor> extractor;

    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if(descriptorType.compare("BRIEF") == 0)
    {
      //cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> extractor;
      extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(64);

    }else if(descriptorType.compare("ORB") == 0)
    {
    //  cv::Ptr<cv::ORB> extractor;
      extractor = cv::ORB::create();
    }else if(descriptorType.compare("FREAK") == 0)
    {
      //cv::Ptr<cv::xfeatures2d::FREAK> extractor;
      extractor = cv::xfeatures2d::FREAK::create();
    }else if(descriptorType.compare("AKAZE") == 0)
    {
      //cv::Ptr<cv::xfeatures2d::FREAK> extractor;
      extractor = cv::AKAZE::create();
    }else
    {
      extractor = cv::xfeatures2d::SIFT::create();
    //cv::Ptr<cv::xfeatures2d::SIFT> detector = cv::xfeatures2d::SIFT::create();
  }
```

## MP.5 Descriptor Matching
<font size="3">
Implement FLANN matching as well as k-nearest neighbor selection. Both methods must be selectable using the respective strings in the main function.
    </font>

## MP.6 Descriptor Distance Ratio
<font size="3">
Use the K-Nearest-Neighbor matching to implement the descriptor distance ratio test, which looks at the ratio of best vs. second-best match to decide whether to keep an associated pair of keypoints.</font>'

---
### FLANN matching
```python
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
      if (descSource.type() != CV_32F || descRef.type() != CV_32F)
      { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
          descSource.convertTo(descSource, CV_32F);
          descRef.convertTo(descRef, CV_32F);
      }
      matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
      cout << "FLANN matching"<<endl;        // ...
    }
```

---
### k-nearest neighbor selection
<font size="3">
Use the K-Nearest-Neighbor matching to implement the descriptor distance ratio test, which looks at the ratio of best vs. second-best match to decide whether to keep an associated pair of keypoints </font>

```python
else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        vector<vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSource, descRef, knn_matches, 2); // finds the 2 best matches
        double minDescDistRatio = 0.8;
        for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
        {
            if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
            {
                matches.push_back((*it)[0]);
            }
        }
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    }
```

## MP.7 Performance Evaluation 1


[//]: # (Image References)

[image1_1]: ./images/SHITOMASI.jpg "SHITOMASI"


<font size="3">
Count the number of keypoints on the preceding vehicle for all 10 images and take note of the distribution of their neighborhood size. Do this for all the detectors you have implemented.</font>


| Detector        | # of Key Points   | 
|:-------------:|:-------------:| 
| SHITOMASI     | 1179        | 
| HARRIS      | 248      |
| SIFT     | 1386      |
| BRISK      | 2762       |
| FAST      | 1491       |
| ORG      | 1161      |
| AKAZE      | 1670      |

### Neighborhood Size

#### SHITOMASI
<figure>
    <img  src="images/SHITOMASI.png" alt="Drawing" style="width: 1000px;"/>
</figure>


#### HARRIS
<figure>
    <img  src="images/Harris.png" alt="Drawing" style="width: 1000px;"/>
</figure>


#### SIFT
<figure>
    <img  src="images/SIFT.png" alt="Drawing" style="width: 1000px;"/>
</figure>


#### FAST
<figure>
    <img  src="images/FAST.png" alt="Drawing" style="width: 1000px;"/>
</figure>


#### BRISK
<figure>
    <img  src="images/BRISK.png" alt="Drawing" style="width: 1000px;"/>
</figure>


#### ORG
<figure>
    <img  src="images/ORG.png" alt="Drawing" style="width: 1000px;"/>
</figure>


#### AKAZE
<figure>
    <img  src="images/AKAZE.png" alt="Drawing" style="width: 1000px;"/>
</figure>

## MP.8 Performance
<font size="3">
Count the number of matched keypoints for all 10 images using all possible combinations of detectors and descriptors. In the matching step, the BF approach is used with the descriptor distance ratio set to 0.8.
    </font>
    
    
   |  Mached Kpts  |SIFT       | ORB      | FREAK    |    BRISK  | BRIEF    |     AKAZE |
   |:-------:|:---------:|:--------:|:--------:|:---------:|:--------:|:---------:|  
   |SHITOMASI| 927       | 908      | 768      | 767       | 944      |  NA   |
   |HARRIS| 163       | 162      | 144      | 142       | 177      |  NA   |
   |SIFT| 800       | NA      | 593      | 899       | 1111      |  NA   |
   |FAST| 1046       | 1071      | 878      | 767       | 944      |  NA   |
   |ORB| 763       | 763      | 420      | 751       | 551      |  NA   |
   |BRISK| 1646       | 1514      | 1524      | 1570       | 1688      |  NA   |
   |AKAZE| 1270       | 1182      | 1187      | 1215       | 1281      |  1259   |

## MP.9 Performance

<font size="3">
Log the time it takes for keypoint detection and descriptor extraction. The results must be entered into a spreadsheet and based on this data, the TOP3 detector / descriptor combinations must be recommended as the best choice for our purpose of detecting keypoints on vehicles.</font>
    
   | Detector|	Descritpor |	Detect Time	| Descript Time	| Time per KPts|
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|     
|	AKAZE	|	AKAZE	|	272.086	|	228.941	|	0.400822
|	AKAZE	|	BRIEF	|	273.191	|	8.10696	|	0.222722
|	AKAZE	|	BRISK	|	279.144	|	11.5211	|	0.231421
|	AKAZE	|	FREAK	|	273.496	|	283.314	|	0.444737
|	AKAZE	|	ORB	|	273.295	|	16.1199	|	0.239582
|	AKAZE	|	SIFT	|	273.093	|	100.46	|	0.292524
|	BRISK	|	BRIEF	|	232.688	|	11.4562	|	0.155506
|	BRISK	|	BRISK	|	232.892	|	18.6075	|	0.157187
|	BRISK	|	FREAK	|	233.081	|	279.919	|	0.343374
|	BRISK	|	ORB	|	231.174	|	25.873	|	0.162483
|	BRISK	|	SIFT	|	233.151	|	134.333	|	0.239248
|	FAST	|	BRIEF	|	5.28216	|	7.36139	|	<font color=red>0.0124078</font>
|	FAST	|	BRISK	|	5.47513	|	10.4666	|	<font color=red>0.0168517</font>
|	FAST	|	FREAK	|	5.32958	|	276.929	|	0.305806
|	FAST	|	ORB	|	5.16305	|	5.46263	|	<font color=red>0.0105101</font>
|	FAST	|	SIFT	|	5.18968	|	80.3707	|	0.084296
|	HARRIS	|	BRIEF	|	77.2746	|	2.4412	|	0.498224
|	HARRIS	|	BRISK	|	81.9919	|	3.30181	|	0.584204
|	HARRIS	|	FREAK	|	78.1611	|	273.532	|	2.31377
|	HARRIS	|	ORB	|	78.4784	|	4.20341	|	0.520011
|	HARRIS	|	SIFT	|	78.6818	|	67.456	|	0.907688
|	ORB	|	BRIEF	|	42.2434	|	5.69279	|	0.0849932
|	ORB	|	BRISK	|	43.3097	|	8.78351	|	0.0726545
|	ORB	|	FREAK	|	41.9896	|	278.602	|	0.776251
|	ORB	|	ORB	|	42.3593	|	25.1892	|	0.0878395
|	ORB	|	SIFT	|	42.7236	|	150.535	|	0.269162
|	SHITOMASI	|	BRIEF	|	64.517	|	5.84294	|	0.077489
|	SHITOMASI	|	BRISK	|	70.0578	|	8.83414	|	0.0959756
|	SHITOMASI	|	FREAK	|	67.8751	|	277.396	|	0.417498
|	SHITOMASI	|	ORB	|	74.2831	|	5.02183	|	0.0881166
|	SHITOMASI	|	SIFT	|	81.1219	|	76.0981	|	0.171825
|	SIFT	|	BRIEF	|	413.827	|	6.89	|	0.55798
|	SIFT	|	BRISK	|	415.306	|	9.86147	|	0.605651
|	SIFT	|	FREAK	|	413.213	|	278.819	|	1.01322
|	SIFT	|	SIFT	|	420.63	|	363.605	|	0.94259


the TOP3 detector / descriptor combinations:

* 1) **FAST** and **ORB**

* 2) **FAST** and **BRIEF**

* 3) **FAST** and **BRISK**
