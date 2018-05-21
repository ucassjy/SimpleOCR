# Implement method: Faster R-CNN with a modified version of RPN
The basic structures of this network are as follows:
### VGG-16 for feature extracting
Convolutional layers of VGG-16 are employed in the front of the framework, which are shared by two sibling branches:

#### RRPN to generate proposals
One branch of the VGG-16 output are sent to RRPN. "The RRPN generates arbitrary-oriented proposals for text instances and further performs bounding box regression for proposals to better fit the text instances." It is followed by two parallel layers respectively doing regression and classification. These two layers form a multi-task loss.

#### RRoI pooling
Output proposals from RRPN are projected onto the VGG-16 output feature map from the other branch. Then RRoI max pooling is employed.

### Final classifier
A classifier consisting of two fully connected layers does the final work, which makes the prediction of whether the proposal has text content.


## Some Understandings of the network
### How does the network make a prediction?
The VGG-16 is used to provide a feature map of the input image. With the feature map the RRPN network makes proposals, which are projected to the feature map itself. Then the following R-CNN network serves as a classifier.

## References:
* Arbitrary-Oriented Scene Text Detection via Rotation Proposals, https://arxiv.org/abs/1703.01086
* Multi-Oriented Scene Text Detection via Corner Localization and Region Segmentation, https://arxiv.org/abs/1802.08948
* RPN, https://blog.csdn.net/wfei101/article/details/78821629
* Faster R-CNN: [Original paper translated into Chinese](https://blog.csdn.net/lwplwf/article/details/74906205); [Discussion 1](https://blog.csdn.net/xbcReal/article/details/76180912), [Discussion 2](https://blog.csdn.net/bailufeiyan/article/details/50575150)
* How to compute IoU for rotated rectangles: [1](https://blog.csdn.net/wfh2015/article/details/78226786)
