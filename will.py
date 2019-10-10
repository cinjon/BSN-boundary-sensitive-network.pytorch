class TEM(torch.nn.Module):
    def __init__(self, opt):
        super(TEM, self).__init__()

        self.nonlinear_factor = opt["tem_nonlinear_factor"]
        self.num_videoframes = opt["num_videoframes"]
        self.out_hidden = opt["tem_hidden_dim"]
        
        key = '%s-%s' % (opt['representation_module'], opt['dataset'])
        # NOTE: The model is the same as `Model` class below.
        model, _, _, representation_dim = _get_module(key)
        self.representation_model = model(opt)
        self.feat_dim = representation_dim
                
        self.conv1 = torch.nn.Conv1d(in_channels=self.feat_dim,
                                     out_channels=self.out_hidden,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     groups=1)
        self.conv2 = torch.nn.Conv1d(in_channels=self.out_hidden,
                                     out_channels=self.out_hidden,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     groups=1)
        self.conv3 = torch.nn.Conv1d(in_channels=self.out_hidden,
                                     out_channels=3,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)

        if opt['tem_reset_params']:
            self.reset_params()

    def _get_representation(self, x):
        """Gets the frozen representation and formats it for downstream.

        Args:
          x: input; Shape is [bs, num_videoframes, ch, h, w]
        """
        with torch.no_grad():
            x = self.representation_model(x)
            
        adj_batch_size = x.shape[0]
        batch_size = int(adj_batch_size / self.num_videoframes)
        x = x.reshape(batch_size, -1, self.num_videoframes)
        return x
            
    def forward(self, x):
        x = self._get_representation(x)
        # NOTE: The x here has shape [bs, 933888, num_videoframes],
        # which is too big for the conv1 right after.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return torch.sigmoid(self.nonlinear_factor * x)


class Model(nn.Module):

    def __init__(self, opts):
        super(Model, self).__init__()
        resnet = resnet_res4s1.resnet50(pretrained=True)
        self.encoderVideo = inflated_resnet.InflatedResNet(
            copy.deepcopy(resnet))

        self.afterconv1 = nn.Conv3d(1024, 512, kernel_size=1, bias=False)

    def forward(self, imgs):
        bs, num_videoframes, ch, h, w = imgs.shape
        imgs = imgs.transpose(1, 2)
        img_feat = self.encoderVideo(imgs)
        # img_feat is now [bs, 1024, num_videoframes, 57, 32]
        img_feat = self.afterconv1(img_feat)
        # img_feat is now [bs, 512, num_videoframes, 57, 32]
        img_feat = img_feat.transpose(1, 2)
        new_shape = [bs * num_videoframes] + list(img_feat.shape[2:])
        img_feat = img_feat.reshape(*new_shape)
        # img_feat is now [bs*num_videoframes, 512, 57, 32]
        return img_feat
