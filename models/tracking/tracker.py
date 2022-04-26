import numpy as np


def getVal():
    return {'keypoints':{ 'nose': [0, 0, 0],
                          'left eye': [0, 0, 0],
                          'right eye': [0, 0, 0],
                          'left ear': [0, 0, 0],
                          'right ear': [0, 0, 0],
                          'left shoulder': [0, 0, 0],
                          'right shoulder':[0, 0, 0],
                          'left elbow': [0, 0, 0],
                          'right elbow': [0, 0, 0],
                          'left wrist': [0, 0, 0],
                          'right wrist': [0, 0, 0],
                          'left hip': [0, 0, 0],
                          'right hip': [0, 0, 0],
                          'left knee': [0, 0, 0],
                          'right knee': [0, 0, 0],
                          'left ankle': [0, 0, 0],
                          'right ankle': [0, 0, 0]
                        }, 
            'bbox': np.array([0, 0, 0, 0]), 
            'score': 0
            }

class Tracker:
    face_items = ['nose', 'left eye', 'right eye', 'left ear', 'right ear']

    def __init__(self) -> None:
        
        self.scores = np.zeros(6)
        self.changeTargetWait= 1
        self.changeTargetThreshold = 0.85
        self.target = getVal()
        self.predictionCoefficient = 0.4
        self.distanceCoefficient = 1
        self.sizeCoefficient = 0.6
        self.counter = 0
        self.previousScore = 0
    
    def get_target_face(self)->list[int]:

        keypoints = self.target['keypoints']
        max_x = 0
        min_x = 1000
        max_y = 0
        min_y = 1000

        for body_part in self.face_items:
            x, y, _ = keypoints[body_part]

            if ( max_x < x):
                max_x = x

            if (min_x>x):
                min_x = x

            if (max_y< y):
                max_y = y

            if (min_y>y):
                min_y = y
        k = 2*(max_y-min_y)        
        min_y = max(0, min_y-k)
        max_y = max(0, max_y+k)
     

        return  [min_x, min_y, max_x, max_y ]


    def ComputeTargetLocation(self, detections:list[dict], /, reset=False,  return_counter=False):
            
            best_score = 0
            best_index = 0
            if self.previousScore == 0 or reset:
                best_index = np.argmax([detection['t_score'] for detection in detections])
                self.target = detections[best_index]
                self.previousScore = best_score
                return self.target

            for index in range(len(detections)):
                score = self.GetScore(detections[index])
                self.scores[index] = score

                if (score > best_score):
                    best_index = index
                    best_score = score

            if ((self.counter > self.changeTargetWait) or (self.previousScore * self.changeTargetThreshold < best_score)):
                self.previousScore = best_score
                self.target = detections[best_index]
                self.counter = 0
                targetIdx = best_index
            else:
                self.counter+=1
            if return_counter:
               return self.target, self.counter  
            else:
              return self.target
                         


    def GetScore(self, detection):
            bbox_score = self.BboxScore(detection['bbox'],
                                  detection['score'])
            keypoint_score = self.GetKeypointScore(detection['keypoints'])

            return bbox_score * keypoint_score

        
    def GetKeypointScore(self, keypoints):
            score = 0
            dist_score = 0

            for  body_part in keypoints:
                x, y, confidence = keypoints[body_part]
                target_x, target_y, _ = keypoints[body_part]
                dist_x = 1 - np.abs(x - target_x)
                dist_y = 1 - np.abs(y - target_y)
                dist_score += (dist_x + dist_y) / 2
                score += confidence

            return (score * self.predictionCoefficient + dist_score * self.distanceCoefficient) / len(keypoints)

    def BboxScore(self, bbox, score):
        xmin, ymin, xmax, ymax = bbox
        size_score = (xmax - xmin) * (ymax - ymin)
        dist_score = self.GetDistanceScore(bbox)
        return self.sizeCoefficient * size_score + dist_score * self.distanceCoefficient + score * self.predictionCoefficient

    
    def GetDistanceScore(self, bbox):
            xmin, ymin, xmax, ymax = bbox
            target_xmin, target_ymin, target_xmax, target_ymax = self.target['bbox']
            dist_xm = CompareDistance(xmax, target_xmax)
            dist_xn = CompareDistance(xmin, target_xmin)
            dist_ym = CompareDistance(ymax, target_ymax)
            dist_yn = CompareDistance(ymin, target_ymin)

            return (dist_xm + dist_xn + dist_ym + dist_yn) / 4


def CompareDistance(x: float, y:float):
        return 1 - np.abs(x - y)


tracker = Tracker()