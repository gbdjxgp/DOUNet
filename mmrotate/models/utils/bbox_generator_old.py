import torch
import numpy as np


class BoxGn:
    def __init__(self):
        # for bbox generator
        self.bbox_offset_matrix = self._get_bbox_offset_matrix().cuda()

    def _get_bbox_offset_matrix(self):
        bbox_offset_matrix = []
        bbox_offset_matrix.append(np.concatenate([[0 for j in range(8)]]))
        bbox_offset_matrix.append(np.concatenate([[0 for j in range(8)]]))
        for k in [4, 2, 1]:
            bbox_offset_matrix.append(np.concatenate(
                [[-1.0 for j in range(k)] + [1.0 for j in range(k)] for i in range(8 // (2 * k))]))
        return torch.Tensor(bbox_offset_matrix)

    def bbox_generator(self, img_labels, img_bboxes, img_metas):
        w = img_bboxes[:, 2]
        h = img_bboxes[:, 3]
        delta = img_bboxes[:, 4]
        # w = img_bboxes[:, 2] - img_bboxes[:, 0]
        # h = img_bboxes[:, 3] - img_bboxes[:, 1]
        # 训练only
        random_offset_matrix = torch.rand(self.bbox_offset_matrix.shape).type_as(img_bboxes)
        random_bbox_offset = 1. / 6 * self.bbox_offset_matrix * random_offset_matrix * torch.stack([torch.zeros_like(w),torch.zeros_like(w),w, h,delta], -1).unsqueeze(-1)




        # random_bbox_offset = self.bbox_offset_matrix * torch.stack([torch.zeros_like(w),torch.zeros_like(w),w, h,delta], -1).unsqueeze(-1)

        proposals = img_bboxes.unsqueeze(-1) + random_bbox_offset
        proposals = proposals.permute([0, 2, 1])

        img_shape = img_metas['img_shape']
        x_, y_, w_, h_,alpha_ = proposals.split([1, 1, 1, 1,1], dim=-1)
        # x1 = x1.clamp(min=0, max=img_shape[1] - 1)
        # y1 = y1.clamp(min=0, max=img_shape[0] - 1)
        # w_ = x2.clamp(min=0, max=img_shape[1] - 1)
        # y2 = y2.clamp(min=0, max=img_shape[0] - 1)
        # 角度处理
        alpha_[alpha_>=torch.pi/2]=alpha_[alpha_>=torch.pi/2]-torch.pi
        alpha_ = alpha_.clamp(min=-torch.pi/2, max=torch.pi/2)
        proposals = torch.cat([x_, y_, w_, h_,alpha_], dim=-1)

        proposal_list = []
        gt_bbox_list = []
        gt_label_list = []
        for label_ind, (label, bbox) in enumerate(zip(img_labels, img_bboxes)):
            # select top sample_number
            proposal = proposals[label_ind]
            gt_bbox = bbox.unsqueeze(0).repeat([len(proposal), 1])
            gt_label = label.new_ones(len(proposal)) * label

            proposal_list.append(proposal)
            gt_bbox_list.append(gt_bbox)
            gt_label_list.append(gt_label)

        img_proposal = torch.cat(proposal_list)
        img_gt_bbox = torch.cat(gt_bbox_list)
        img_gt_label = torch.cat(gt_label_list)

        return img_proposal, img_gt_bbox, img_gt_label
