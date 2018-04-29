import numpy as np
import cupy as cp
from .utils.nms import non_maximum_suppression
from torch import nn

class FasterRCNN(nn.Module):
    """Base class for Faster R-CNN.

        This is a base class for Faster R-CNN links supporting object detection
        API [#]_. The following three stages constitute Faster R-CNN.

        1. **Feature extraction**: Images are taken and their \
            feature maps are calculated.
        2. **Region Proposal Networks**: Given the feature maps calculated in \
            the previous stage, produce set of RoIs around objects.
        3. **Localization and Classification Heads**: Using feature maps that \
            belong to the proposed RoIs, classify the categories of the objects \
            in the RoIs and improve localizations.

        Args:
            extractor (nn.Module): A module that takes a BCHW image
                array and returns feature maps.
            rpn (nn.Module): A module that has the same interface as
                :class:`model.region_proposal_network.RegionProposalNetwork`.
                Please refer to the documentation found there.
            head (nn.Module): A module that takes
                a BCHW variable, RoIs and batch indices for RoIs. This returns class
                dependent localization paramters and class scores.
            loc_normalize_mean (tuple of four floats): Mean values of
                localization estimates.
            loc_normalize_std (tupler of four floats): Standard deviation
                of localization estimates.

        """
    def __init__(self, extractor, rpn, head,
                loc_normalize_mean = (0., 0., 0., 0.),
                loc_normalize_std = (0.1, 0.1, 0.2, 0.2)):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head

        # mean and std
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.use_preset('evaluate')

    @property
    def n_class(self):
        # Total number of classes including the background.
        return self.head.n_class

    def forward(self, x, scale=1.):
        """Forward Faster R-CNN.
            Args:
                x (autograd.Variable): 4D image variable.
                scale (float): the threshold used to select small object

            Returns:
                Variable, Variable, array, array:
                Returns tuple of four values listed below.

                * **roi_cls_locs**: Offsets and scalings for the proposed RoIs. \
                        Its shape is :math:`(R', (L + 1) \\times 4)`.
                * **roi_scores**: Class predictions for the proposed RoIs. \
                        Its shape is :math:`(R', L + 1)`.
                * **rois**: RoIs proposed by RPN. Its shape is \
                        :math:`(R', 4)`.
                * **roi_indices**: Batch indices of RoIs. Its shape is \
                        :math:`(R',)`.
        """
        img_size = x.shape[2:]

        feat = self.extractor(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(feat, img_size, scale)
        roi_cls_locs, roi_scores = self.head(feat, rois, roi_indices)
        return roi_cls_locs, roi_scores, rois, roi_indices

    def use_preset(self, preset):
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        else:
            raise ValueError('preset must be visualize or evaluate')

    def _suppress(self, raw_cls_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()
        # skip cls_id = 0 because it is the background class
        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            keep = non_maximum_suppression(cp.array(cls_bbox_l), self.nms_thresh, prob_l)
            keep = cp.asnumpy(keep)
            bbox.append(cls_bbox_l[keep])
            # The labels are in [0, self.n_class - 2].
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep])
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score





