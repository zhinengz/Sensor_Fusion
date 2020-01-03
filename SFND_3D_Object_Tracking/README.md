# Track an Object in 3D Space

## TASK.1 Match 3D Objects
##  TASK.3 Associate Keypoint Correspondences with Bounding Boxes


    Task FP.1 and FP.3 are both implemented in the function of matchBoundingBoxes()


---

### TASK.1:    

Implement the method **matchBoundingBoxes()**, which takes as input both the previous and the current data frames and provides as output the ids of the matched regions of interest (i.e. the boxID property). Matches must be the ones with the highest number of keypoint correspondences.

---
**Steps:**
1. Loop through all the bounding boxes in the current frame
2. Loop through all the matching points, and check wthether the matching point is within the current box. If it is, the corresponding box ID in the previous frame, which includes the matched point, is checked and counted.
3. After step 1 and 2, each box in the current frame is matched to 0, 1 or multiple matched boxes from the previous frame, the box having the highest number of keypoint correspondences is picked as the final matched one for the current box 
4. After boxes are matched, the associate keypoints and kptMatches are re-assigned to both boxes (Task FP.3)

    matching points: keypoints[it2->trainIdx]    
    matched points: keypoints[it2->queryIdx]

---

**matchBoundingBoxes():**  
_Loop through all the bounding boxes in the current frame_

```C++
for(auto it1 = currFrame.boundingBoxes.begin(); it1!=currFrame.boundingBoxes.end(); ++it1)
    {              
      map<int,int> tmpMatchedBoxes; //map<pre_BoxID, # of matched KeyPoints in pre_Box>
```
_Loop through all the matching points_


```C++      
      for(auto it2 = matches.begin(); it2!=matches.end(); ++it2)  //loop through all matched points
      {
        cv::KeyPoint currPt =currFrame.keypoints[it2->trainIdx] ;
```
_If current box contains this matching point currPt, the boxID containing matched point prevPt in the previous frame needs to be found out as showing below_
       
```C++       
        if(it1->roi.contains(currPt.pt)){
          cv::KeyPoint prevPt =prevFrame.keypoints[it2->queryIdx] ;
          for(auto it3 = prevFrame.boundingBoxes.begin();
                   it3!=prevFrame.boundingBoxes.end();++it3)
          {

```
_If the box in the previous frame contains the matched point prevPt, record this box ID and how many times it has matched points_       
```C++               
            if(it3->roi.contains(prevPt.pt))
            {
              int boxId = it3->boxID;
              auto it_tmp = tmpMatchedBoxes.find(boxId);
              if(it_tmp==tmpMatchedBoxes.end())
              {tmpMatchedBoxes.insert(make_pair(boxId, 1));}
              else{
                tmpMatchedBoxes[boxId] +=1;
              }
            }
          }
        }
      }
```
_Now each boxes in the current frame could have 0, 1 or multiple matched boxes from the previous frame,the box having the highest number of keypoint correspondences is picked as the final matching box for the current box_       
```C++  
    int bestMatchBox = -1;
    int bestMatchValue = 0;
    for(auto it=tmpMatchedBoxes.begin();it!=tmpMatchedBoxes.end();++it)
    {
      if(it->second>bestMatchValue)
      {
        bestMatchBox = it->first;
        bestMatchValue = it->second;
      }
    }
      bbBestMatches.insert(pair<int,int>(bestMatchBox,it1->boxID));
```


---
### TASK.3:

Prepare the TTC computation based on camera measurements by associating keypoint correspondences to the bounding boxes which enclose them. All matches which satisfy this condition must be added to a vector in the respective bounding box.
    
---

As mentioned in FP.1, matches between boxes are defined as the ones with the highest number of keypoint correspondences, meaning some keypoints contained by a bounding box in the current frame are not included in its corresponding box in the previous frame. The following figure shows such a case: some keypoints pointed by the red arrow are enclosed by the current box, but not included by its corresponding box in the previous frame:

<figure>
    <img  src="images/fig1.png" alt="Drawing" style="width: 1000px;"/>
</figure>

Those matched keypoints not included by both boxes need to be removed.


```C++
      std::vector<BoundingBox>::iterator it3;
      for( it3 = prevFrame.boundingBoxes.begin(); it3 != prevFrame.boundingBoxes.end();++it3)
      {
        if(it3->boxID == bestMatchBox)  //to find out bounding box in the previous frame based on boxID
        break;
      }
      it1->kptMatches.clear(); 
      it1->keypoints.clear();
      it3->keypoints.clear();
      for(auto it2 = matches.begin();it2!=matches.end();++it2)
      {
        cv::KeyPoint currPt =currFrame.keypoints[it2->trainIdx] ;
        cv::KeyPoint prevPt =prevFrame.keypoints[it2->queryIdx] ;
```
_Only record those matched points contained by both boxes_
```C++
        if(it1->roi.contains(currPt.pt)  && it3->roi.contains(prevPt.pt))
        {
          it1->kptMatches.push_back(*it2);
          it1->keypoints.push_back(currPt);
          it3->keypoints.push_back(prevPt);
        }
      }
```

The following figure shows improved keypoint correspondences of figure 1
<figure>
    <img  src="images/fig2.png" alt="Drawing" style="width: 1000px;"/>
</figure>

The following figure shows keypoint correspondences for the bounding boxes:

<figure>
    <img  src="images/boxmatch.gif" alt="Drawing" style="width: 2000px;"/>
</figure>



---
## TASK.2 Compute Lidar-based TTC

---
Compute the time-to-collision in second for all matched 3D objects using only Lidar measurements from the matched bounding boxes between current and previous frame. Find a method to handle those outliers

---
For this task, I changed **LidarPoint** struct by overloading "<" operator, so that LiarPoint will be sorted based on X value.

```C++
struct LidarPoint { // single lidar point in space
    double x,y,z,r; // x,y,z in [m], r is point reflectivity
    bool operator < (const LidarPoint& pt) const
    {
      return x < pt.x;
    }
};

```

Function **computeTTCLidar()** become much simpler by overloading "<" operator for LidarPoint struct, after sorting lidar point, **lidarPointsPrev.begin()** is the element closest to Lidar sensor

```C++
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    double minXPrev = 1e9, minXCurr = 1e9;
    minXPrev = lidarPointsPrev.begin()->x;
    minXCurr = lidarPointsCurr.begin()->x;
    TTC = minXCurr  / (minXPrev - minXCurr+1e-6)/frameRate;
}
```

### Handling Outliers

3. In the following figure, I highlight the first outlier point in red. With sorting of lidar points, those outliers are located at the beginning part of lidarpoint vector. 

<figure>
    <img  src="images/fig4.png" alt="Drawing" style="width: 800px;"/>
</figure>

The way taken to remove outliers is  to use the **IQR range values**. That is, all the lidarpoints with x value less than **Q1 - 1.5*IQR** are removed.

```C++
void removeOutlier(std::vector<LidarPoint> &lidarPoints)
{
  sort(lidarPoints.begin(),lidarPoints.end());
  long medIdx = floor(lidarPoints.size()/2.0);
  long medianPtindex = lidarPoints.size()%2==0?medIdx -1 :medIdx;

  long Q1Idx = floor(medianPtindex/2.0);
  double Q1 = medianPtindex%2==0?(lidarPoints[Q1Idx -1].x +lidarPoints[Q1Idx].x)/2.0:lidarPoints[Q1Idx].x;
  auto it=lidarPoints.begin();
  while(it!=lidarPoints.end())
  {
    if(it->x < Q1)
    {
      it = lidarPoints.erase(it);
    }else{break;}
  }
}

```

The following figure shows the outlier appeared in the above figure is removed and the new nearest lidar point location is changed

<figure>
    <img  src="images/fig41.png" alt="Drawing" style="width: 800px;"/>
</figure>


### Lidar-based TTC:


<table><tr>
<td><figure>
    <img  src="images/ttc_lidar_with_outlier.png" alt="Drawing" style="width: 500px;" />
    <figcaption>With Outliers</figcaption>
    </figure></td>
<td><figure>
    <img  src="images/ttc_lidar_without_outlier.png" alt="Drawing" style="width: 500px;"/>     
    <figcaption>Outliers removed</figcaption></figure></td>
</tr></table>



The following animation shows the lidar point projected to 2D image with nearest lidar point highlighted in red. It can be observed that the nearest lidar point location is **not** fixe, but moving around, which can cause lidar-based TTC to have big variation and even outliers 

<figure>
        <img  src="images/TTC.gif" alt="Drawing" style="width: 800px;"/>    
</figure>

---
## TASK.4 Compute Camera-based TTC
Compute the time-to-collision in second for all matched 3D objects using only keypoint correspondences from the matched bounding boxes between current and previous frame.

---
### Deal with outlier correspondences
Both Harris and ORB detectors shows much more outliers than other keypoint detector:

#### Before improvement:

**HARRIS:**

<figure>
    <img  src="images/ttc_Harris_100.png" alt="Drawing" style="width: 500px;"/>
</figure>

**ORB:**
<figure>
    <img  src="images/ttc_ORB_HARRIS.png" alt="Drawing" style="width: 500px;"/>
</figure>

The reason of so many outliers is due to no enough keypoints been detected within the bounding box. The following shows the matched keypoints from the front car using ORB detector with default setting
<figure>
    <img  src="images/ORB_Harris.png" alt="Drawing" style="width: 1000px;"/>
</figure>

ORB detector has been updated by setting FAST_SCORE to rank feature and set nFeatures to **7000** to increase keypoints 
```C++
double detKeypointsORB(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
  int nFeatures = 7000;
  float scaleFactor = 1.2f;
  int nLevels = 8;
  int edgeThreshold = 31;
  int firstLevel = 0;
  int WTA_K = 2;
  int scoreType = ORB::FAST_SCORE;

  cv::Ptr<cv::ORB> detector = cv::ORB::create(nFeatures, scaleFactor, nLevels,edgeThreshold,firstLevel,WTA_K,cv::ORB::FAST_SCORE);
```
The following figure shows improved matched keypoints using new feature ranking algorithm

<figure>
    <img  src="images/ORB_FAST_SCORE.png" alt="Drawing" style="width: 1000px;"/>
</figure>

For HARRIS detector, the minResponse value is changed from 100 to 80 for NMS algorithm, but the computation time is increased a lot.

#### After improvement:
Outliers for ORB and HARRIS are both reduced considerably

**HARRIS:**
<figure>
    <img  src="images/ttc_Harris_80.png" alt="Drawing" style="width: 500px;"/>
</figure>

**ORB:**

<figure>
    <img  src="images/ttc_ORB_FAST_SCORE.png" alt="Drawing" style="width: 500px;"/>
</figure>


#### Camera-based TTC for all other keypoint detectors:


<figure>
    <img  src="images/Camera_based_TTC.png" alt="Drawing" style="width: 1000px;"/>
</figure>


---
## TASK.5 Performance Evaluation of Lidar-based TTC

---
Find examples where the TTC estimate of the Lidar sensor does not seem plausible. Describe your observations and provide a sound argumentation why you think this happened.

---
The following figure shows lidar-based TTC outliers for freame 6 and 7. The cause could be found where the nearest lidar point is located for frame 6 and 7

<figure>
    <img  src="images/ttc_lidar.png" alt="Drawing" style="width: 500px;"/>
</figure>

The rear of the preceding vehicle is not flat, different locations have different distance to the lidar sensor, intuitively:
1. the top-right corner of bump is probably the cloest location to lidar sensor
2. the bottom part, or the location where car plate is mounted, is most far from the lidar sensor


The following 3 figures show the nearest lidar points locations for 3 consecutive frames

The first figure shows the hot lidar point is located at the bumpers top-right corner 

<figure>
    <img  src="images/img5.png" alt="Drawing" style="width: 1000px;"/>
</figure>

The second figure shows the hot lidar point is moved to the top of car plate, the depth where plate is mounted makes the predicted car distance bigger than real value, given a smaller TTC (~7S).

<figure>
    <img  src="images/img6.png" alt="Drawing" style="width: 1000px;"/>
</figure>

The third figure shows the hot lidar point is still on the top of car plate, the overestimated car distance from the previous frame make the predicted car distance change smaller than the real value, given a bigger TTC (~34S).
<figure>
    <img  src="images/img7.png" alt="Drawing" style="width: 1000px;"/>
</figure>

The above 3 figure shows 3 consecutive nearest lidar point, the hot lidar point is at top-right bumper corner, 

---
## FP.6 : Performance Evaluation Camera-based TTC

Run several detector / descriptor combinations and look at the differences in TTC estimation. Find out which methods perform best and also include several examples where camera-based TTC estimation is way off. As with Lidar, describe your observations again and also look into potential reasons.

---


#### Best Camera-based TTC:


<figure>
    <img  src="images/Camera_based_TTC_best.png" alt="Drawing" style="width: 1000px;"/>
</figure>

Both HARRIS, ORB and BRISK shows some way-off estimation  as shown in **FP.4**

Lidar-based TTC show comparable performance as camera-based, because lidar sensor directly measure distance
