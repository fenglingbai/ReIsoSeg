#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from torch import nn


class MultipleOutputLoss2(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss2, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, x, y):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        l = weights[0] * self.loss(x[0], y[0])
        for i in range(1, len(x)):
            if weights[i] != 0:
                l += weights[i] * self.loss(x[i], y[i])
        return l


class ReIsoLossHistory(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(ReIsoLossHistory, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss
        self.ani_loss_list = []
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # # print(device)
        # # 将tensor传送到设备
        # self.weight1 = self.weight1.to(device)
        # self.weight2 = self.weight2.to(device)

    def forward(self, x, y, ani_scale):
        assert ani_scale >= 0
        assert ani_scale <= 1
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        loss_list = [weights[0] * self.loss(x[0], y[0])]
        for i in range(1, len(x)):
            if weights[i] != 0:
                loss_list.append(weights[i] * self.loss(x[i], y[i])) 
        # iso
        l_iso = loss_list[0]
        # ani
        l_ani = 0
        if len(x) % 2 == 0:
            for i in range(1, len(x)//2):
                l_iso = l_iso + loss_list[i]
            for i in range(len(x)//2, len(x)):
                l_ani = l_ani + loss_list[i]
        # print('l_iso: ', l_iso, ' l_ani: ', l_ani)
        l = (1 - ani_scale) * l_iso+ ani_scale * l_ani
        return l, l_iso, l_ani