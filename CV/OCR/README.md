# CRAFT (Character Region Awareness For Text Detection)

## Architecture
  
  - VGG16 wtih Batch Normalization is adopted as backbone
  - also adopted Fully Convolutional Networks (FCN)
  - U-net like architecture is used to reflect both shallow and deep level features
  - <==> Feature extraction + Localization
## Learning

### Objective
Pixel-wise prediction of **region score and affinity score** on maps.

<img width="702" alt="스크린샷 2023-05-09 오후 4 36 40" src="https://github.com/1nilx2/Deep-Learning/assets/88100984/e8c6711a-752d-4634-8083-2199eba6ac57">

Region score = P(center of a character)  
Affinity score = P(center of adjacent two chars), which will be used to combine chars to word

Both scores can be expressed by isotropic gaussian map 

### Loss

<img width="803" alt="스크린샷 2023-05-09 오후 5 14 32" src="https://github.com/1nilx2/Deep-Learning/assets/88100984/59b23700-f065-49dc-b783-973d8721a3df">

Loss conists of two parts
  - Finding Euclidean distances for region score and affinity score
  - Multlying confidence score of the pixel (Mainly related with Pseudo-Ground Truth)


### Training
Character-level dataset is limited, and manual labeling is too costly.  
So we
 1) Train model with Ground Truth <- Interim Model
 2) Generates Pseudo GT with confidence through Interim Model
 3) Conduct additional learning with PGT (Weekly-Supervised Learning)


Really good references
  - https://medium.com/@msmapark2/character-region-awareness-for-text-detection-craft-paper-%EB%B6%84%EC%84%9D-da987b32609c
  - https://towardsdatascience.com/pytorch-scene-text-detection-and-recognition-by-craft-and-a-four-stage-network-ec814d39db05
