from easyai.name_manager.loss_name import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.utility.loss_registry import REGISTERED_CLS_LOSS


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        class_number (int): number of classes.
        feature_dim (int): feature dimension.
    """

    def __init__(self, class_number=10, feature_dim=2):
        super(CenterLoss, self).__init__()
        self.class_number = class_number
        self.feature_dim = feature_dim
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

        nn.init.normal_(self.centers, mean=0, std=1)

    def forward(self, input_data, labels):
        """
        Args:
            input_data: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        device = input_data.device
        batch_size = input_data.size(0)
        distmat = torch.pow(input_data, 2).sum(dim=1, keepdim=True).\
                      expand(batch_size, self.class_number) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).\
                      expand(self.class_number, batch_size).t()
        distmat.addmm_(1, -2, input_data, self.centers.t().to(device))

        classes = torch.arange(self.class_number).long().to(device)
        labels = labels.unsqueeze(1).expand(batch_size, self.class_number)
        mask = labels.eq(classes.expand(batch_size, self.class_number))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss
