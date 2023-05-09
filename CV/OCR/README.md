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

$L = \sum_p{\|\|S_r(p) - S_{r}^{*}(p)\|\|^{2}_{2} + \|\|S_a(p) - S_a^*(p)\|\|^2_2}$


  
