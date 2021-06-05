class SegBranch(nn.Module):
	def __init__(self, net_enc, net_dec):
		super(SegBranch, self).__init__()
		self.encoder = net_enc
		self.decoder = net_dec

	def forward(self, data):
		feats = self.encoder(data, return_feature_maps=True)
		pred = self.decoder(feats)
		if isinstance(pred,list):
			for i in range(len(pred)):
				pred[i] = F.upsample(pred[i], size=data.size()[2:], mode='bilinear', align_corners=False)
		else:
			pred = F.upsample(pred, size=data.size()[2:], mode='bilinear', align_corners=False)
      
		return pred
  
class SegDecoder(nn.Module):
	def __init__(self, num_class=1, fc_dim=2048, use_softmax=False, pool_scales=(1, 2, 3, 6), fpn_inplanes=(256,512,1024,2048), fpn_dim=256,freez_bn=True):
		super(SegDecoder, self).__init__()
		self.use_softmax = use_softmax

		# PPM Module
		self.ppm_pooling = []
		self.ppm_conv = []

		for scale in pool_scales:
			self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
			self.ppm_conv.append(nn.Sequential(
				nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
				nn.BatchNorm2d(512),
				nn.ReLU(inplace=True)
			))
		self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
		self.ppm_conv = nn.ModuleList(self.ppm_conv)
		self.ppm_last_conv = ConvBatchNormReLU(fc_dim + len(pool_scales)*512, fpn_dim)

		# FPN Module
		self.fpn_in = []
		for fpn_inplane in fpn_inplanes[:-1]: # skip the top layer
			self.fpn_in.append(nn.Sequential(
				nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
				nn.BatchNorm2d(fpn_dim),
				nn.ReLU(inplace=True)
			))
		self.fpn_in = nn.ModuleList(self.fpn_in)

		self.fpn_out = []
		for i in range(len(fpn_inplanes) - 1): # skip the top layer
			self.fpn_out.append(nn.Sequential(
				ConvBatchNormReLU(fpn_dim, fpn_dim)
			))
		self.fpn_out = nn.ModuleList(self.fpn_out)

		self.score_out = []
		for i in range(len(fpn_inplanes)):  # skip the top layer
			self.score_out.append(nn.Sequential(
				ConvBatchNormReLU(fpn_dim, fpn_dim),
				nn.Conv2d(fpn_dim, num_class, 1),
			))
		self.score_out = nn.ModuleList(self.score_out)

		self.upscale = []
		for i in range(len(fpn_inplanes) - 1):
			self.upscale.append(nn.ConvTranspose2d(fpn_dim, fpn_dim, kernel_size=4, stride=2, bias=False))
		self.upscale = nn.ModuleList(self.upscale)
		self.att_out = []
		for i in range(len(fpn_inplanes)-1):  # skip the top layer
			self.att_out.append(Refine(fpn_dim, 1))
		self.att_out = nn.ModuleList(self.att_out)

		self.conv_last = nn.Sequential(
			ConvBatchNormReLU(len(fpn_inplanes) * fpn_dim, fpn_dim),
			nn.Conv2d(fpn_dim, num_class, kernel_size=1)
		  )

		if freez_bn == True:
			self.freeze_bn()

	def forward(self, conv_out, segSize=None):
		results = []
		conv5 = conv_out[-1]
		input_size = conv5.size()
		ppm_out = [conv5]
		for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
			ppm_out.append(pool_conv(nn.functional.upsample(
				pool_scale(conv5),
				(input_size[2], input_size[3]),
				mode='bilinear', align_corners=False)))
		ppm_out = torch.cat(ppm_out, 1)
		f = self.ppm_last_conv(ppm_out)

		seg_res = self.score_out[-1](f)
		# seg_res_up = F.upsample(seg_res, size=conv_out[0].size()[2:], mode='bilinear', align_corners=False)
		results.append(seg_res)

		fpn_feature_list = [f]
		for i in reversed(range(len(conv_out) - 1)):
			conv_x = conv_out[i]
			conv_x = self.fpn_in[i](conv_x) # lateral branch

			# f = F.upsample(f, size=conv_x.size()[2:], mode='bilinear', align_corners=False) # top-down branch
			f = crop_like(self.upscale[i](f), conv_x)
			f = conv_x + f
			f_1 = self.fpn_out[i](f)

			seg_res = F.upsample(seg_res, size=conv_x.size()[2:], mode='bilinear', align_corners=False)
			seg_res = F.sigmoid(seg_res)
			f_1 = self.att_out[i]([f_1, seg_res])
			seg_res = self.score_out[i](f_1)
			# seg_res_up = F.upsample(seg_res, size=conv_out[0].size()[2:], mode='bilinear', align_corners=False)
			results.append(seg_res)

			fpn_feature_list.append(f_1)

		fpn_feature_list.reverse() # [P2 - P5]
		output_size = fpn_feature_list[0].size()[2:]
		fusion_list = [fpn_feature_list[0]]
		for i in range(1, len(fpn_feature_list)):
			fusion_list.append(nn.functional.upsample(
				fpn_feature_list[i],
				output_size,
				mode='bilinear', align_corners=False))
		fusion_out = torch.cat(fusion_list, 1)
		x = self.conv_last(fusion_out)
		results.append(x)

		return results

	def freeze_bn(self):
		for m in self.modules():
			if isinstance(m, nn.BatchNorm2d):
				m.eval()  
        
class SegEncoder(nn.Module):
	def __init__(self, multi_grid=[1, 2, 1],freez_bn=True):
		self.inplanes = 64
		layers = [3, 4, 23, 3]
		block = Bottleneck
		super(SegEncoder, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)

		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=1,dilation=2)
		self.layer4 = self._make_layer_mg(block, 512, layers[3], stride=1, dilation=2, mg=multi_grid)

		self._initialize_weights()
		if freez_bn == True:
			self.freeze_bn()

	def forward(self, x, return_feature_maps=False):
		conv_out = []
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x); conv_out.append(x);
		x = self.layer2(x); conv_out.append(x);
		x = self.layer3(x); conv_out.append(x);
		x = self.layer4(x); conv_out.append(x);

		if return_feature_maps:
			return conv_out
		return [x]

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				m.weight.data.normal_(0, 0.01)
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()

	def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
			downsample = nn.Sequential(
			nn.Conv2d(self.inplanes, planes * block.expansion,
					kernel_size=1, stride=stride, bias=False),
			nn.BatchNorm2d(planes * block.expansion)
			)
		layers = []
		layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes, dilation=dilation))

		return nn.Sequential(*layers)

	def _make_layer_mg(self, block, planes, blocks=3, stride=1, dilation=2, mg=[1, 2, 1]):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion)
			  )
		layers = []
		layers.append(block(self.inplanes, planes, stride, dilation=dilation*mg[0], downsample=downsample))
		self.inplanes = planes * block.expansion
		layers.append(block(self.inplanes, planes, dilation=dilation*mg[1]))
		layers.append(block(self.inplanes, planes, dilation=dilation*mg[2]))
		return nn.Sequential(*layers)

	def freeze_bn(self):
		for m in self.modules():
			if isinstance(m, nn.BatchNorm2d):
				m.eval()
