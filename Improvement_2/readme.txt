All the files listed below are variations of the  second improvement 

For the classifier.ipynb, we use YOLOv8s for precise fire localization and CLIP for semantic verification, 
combining their outputs via a logistic classifier head or direct fusion. The classifier is fine-tuned to 
balance YOLO’s detection confidence and CLIP’s fire probability, using weights alpha=0.6 and beta=0.4. 

For the padding.ipynb, the only difference is that when we train the classifier we use padding=0.1 
so that more information is retained.

For the prompting.ipynb, the only modification we have is that the prompts used for evaluating the 
classifier have the same categories, but are shorter in length and a bit simpler

For the hyperparameter.ipynb, the only modification is that we use a higher beta value than alpha 
to increse the influence the CLIP model.