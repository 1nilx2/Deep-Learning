# Metrics for Segmentation
 
## Accuracy
 We can count the number of pixels predicted correctly among the total number of pixels.
 $$Accuracy(A, B) = \frac{1}{n} \Sigma^{n}_{i=0}{1_{(A_i=B_i)}}$$
 This works well when our class distribution is rather balanced. 
 What if our image has 1,000 total pixels and only 3 of them is for an object?
 If our model predicts entire pixels as backgound, it will still have 0.997 accuracy.  
This is somewhat exaggerated, but usually images consist of less pixels for objects than those for background.

## Precision and Recall
 Thus, we might want to speculate more detailed traits of our model. How precisely a model captured truth among what it predicted or how much truth it has captured among total truth. The former is $Precision$ and the later is $Recall$.
 $$Precision = \frac{TP}{TP+FP}$$
 $$Recall = \frac{TP}{TP+FN}$$

## F1 Score & IoU
### $Dice (F1)$
If you want to see these values at once with a condensed figure, F1 score can be used, which is minimized when either has 0 and maximized when both have 1. Also F1 score is menteiond as $Dice \ Score$.
$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$
$$F1 = 2 * \frac{\frac{TP^2}{(TP+FP)(TP+FN)}}{TP(\frac{2TP+FP+FN}{(TP+FP)(TP+FN)})}$$
$$F1 = \frac{2TP}{2TP+FP+FN}$$

### $Jaccard (IoU)$
Intersection over Union (IoU) is another metric that can represent a segmentation performance. 
$$IoU = \frac{TP}{TP+FP+FN}$$

## mAP
Mean average Precision.
Mean for various classes, and Average for varying recall levels. As a measurement for capturing both $Precision$ and $Recall$, we can use $PR \ AUC$, area under the Precision-Recall curve, which can shows how $Precision$ differs according to the increase in $Recall$. This can be calculated for each class, then if averaged, that will be mAP. 

As you know, a model's confusion matrix composition can be different, depending on the threshhold you set. For segmentations, IoU is used as the threshold. Thus, mAP at IoU 50% can have different result compared to mAP at IoU 95%. We call each $mAP_{50}$ and $mAP_{95}$. Especially in segmentation setting, you'd also want to see balanced result from $mAP_{50}, mAP_{55}, ... , mAP_{95}$, which can be simply expressed as $mAP$.