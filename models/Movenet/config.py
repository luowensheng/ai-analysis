
KEYPOINT_NAMES = [  "nose", "left eye", "right eye", "left ear", "right ear", "left shoulder", 
                    "right shoulder", "left elbow", "right elbow", "left wrist", "right wrist", 
                    "left hip", "right hip", "left knee", "right knee", "left ankle", "right ankle"
                ]
                
CONNECTIONS = [
    # face
    ["right ear", "right eye"],
    ["right eye", "nose"],
    ["nose", "left eye"],
    ["left eye", "left ear"],
    # mid
    ["left shoulder", "right shoulder"],
    # left side
    ["left shoulder", "left elbow"],
    ["left elbow", "left wrist"],
    # right side
    ["right shoulder", "right elbow"],
    ["right elbow", "right wrist"],
    # hip
    ["left shoulder", "left hip"],
    ["right shoulder", "right hip"],
    # right leg
    ["right hip", "right knee"],
    ["right knee", "right ankle"],
    # left leg
    ["left hip", "right knee"],
    ["left knee", "left ankle"],

]