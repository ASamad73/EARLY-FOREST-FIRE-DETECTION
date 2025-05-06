The first model (model 1) we made involved the analysis of fire imagery using a combination of YOLOv8 and CLIP. We utilized two distinct datasets: one comprising general fire scenes and the other focusing on forest fires captured from a distance.

Our initial experiments, where YOLOv8 was employed for fire detection followed by CLIP for image analysis, revealed a performance disparity. The YOLOv8 model demonstrated higher accuracy metrics when processing general fire images. In contrast, the distant forest fire images, characterized by subtle visual cues and variations in scale and perspective, presented a greater challenge for accurate detection by YOLOv8.

General Fire when tested on model 1 showed better performance than when Forest Fire Dataset was used.

Subsequently, in our second model (model 2), in accordance with the professor's guidance, we integrated YOLOv8 and CLIP to perform joint inferences. This refined approach resulted in a notable improvement in the model's performance, particularly with the forest fire dataset. The combined methodology allowed for more effective identification and classification of these previously challenging images.