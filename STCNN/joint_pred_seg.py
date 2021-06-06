class FramePredEncoder(nn.Module):
	def __init__(self,frame_nums=4):
		self.inplanes = 64
		layers = [3, 4, 23, 3]
		block = Bottleneck
		super(FramePredEncoder, self).__init__()
		self.conv1 = nn.Conv2d(frame_nums*3, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		# maxpool different from pytorch-resnet, to match tf-faster-rcnn
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

		self._initialize_weights()

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()
			elif isinstance(m, nn.ConvTranspose2d):
				m.weight.data.zero_()
				m.weight.data = interp_surgery(m)

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
			  nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
			  nn.BatchNorm2d(planes * block.expansion)
			)
		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample=downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

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

class FramePredDecoder(nn.Module):
	def __init__(self):
		super(FramePredDecoder, self).__init__()
		# Decoder
		self.convC_1 = nn.Conv2d(512 * 4, 512 * 2, kernel_size=1, stride=1)
		self.convC_2 = nn.Conv2d(512 * 2, 512, kernel_size=1, stride=1)
		self.convC_3 = nn.Conv2d(512, 256, kernel_size=1, stride=1)

		self.de_layer1 = nn.Sequential(nn.Conv2d(512 * 2, 512, kernel_size=3, stride=1, padding=1),
					       nn.BatchNorm2d(512),
					       # nn.ReLU(inplace=True),
					       nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, bias=False)
					      )

		self.de_layer2 = nn.Sequential(nn.Conv2d(512 * 2, 256, kernel_size=3, stride=1, padding=1),
					       nn.BatchNorm2d(256),
					       # nn.ReLU(inplace=True),
					       nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, bias=False)
					      )

		self.de_layer3 = nn.Sequential(nn.Conv2d(256 * 2, 64, kernel_size=3, stride=1, padding=1),
					       nn.BatchNorm2d(64),
					       # nn.ReLU(inplace=True),
					       nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, bias=False),
					       nn.Conv2d(64, 3, kernel_size=1, stride=1)
					      )
		self._initialize_weights()

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()
			elif isinstance(m, nn.ConvTranspose2d):
				m.weight.data.zero_()
				m.weight.data = interp_surgery(m)

	def forward(self, conv_feats, return_feature_maps=False):
		conv_out = []
		x4 = self.convC_1(conv_feats[-1])
		out1 = self.de_layer1(x4)
		out1 = crop_like(out1, conv_feats[-2]);conv_out.append(out1);
		x3 = self.convC_2(conv_feats[-2])
		out2 = self.de_layer2(torch.cat((out1, x3), 1))
		out2 = crop_like(out2, conv_feats[-3]);conv_out.append(out2);
		x2 = self.convC_3(conv_feats[-3])
		out3 = torch.cat((out2, x2), 1)
		modulelist = list(self.de_layer3.modules())
		for l in modulelist[1:-1]:
			out3 = l(out3)
		out3 = crop_like(out3, conv_feats[-4]);conv_out.append(out3);
		out4 = modulelist[-1](out3)
		pred = F.tanh(out4)

		if return_feature_maps:
			return pred, conv_out
		
		return pred

class JointSegDecoder(nn.Module):
	def __init__(self, num_class=1, fc_dim=2048, pool_scales=(1, 2, 3, 6),
				 fpn_inplanes=(256,512,1024,2048), fpn_dim=256,freez_bn=True):
		super(JointSegDecoder, self).__init__()

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

		fpn_out = []
		fpn_out.append(nn.Sequential(ConvBatchNormReLU(fpn_dim+64, fpn_dim)))
		fpn_out.append(nn.Sequential(ConvBatchNormReLU(fpn_dim+256, fpn_dim)))
		fpn_out.append(nn.Sequential(ConvBatchNormReLU(fpn_dim+512, fpn_dim)))
		self.joint_fpn_out = nn.ModuleList(fpn_out)

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
		for i in range(len(fpn_inplanes) - 1):  # skip the top layer
			self.att_out.append(Refine(fpn_dim, 1))
		self.att_out = nn.ModuleList(self.att_out)

		self.conv_last = nn.Sequential(
			ConvBatchNormReLU(len(fpn_inplanes) * fpn_dim, fpn_dim),
			nn.Conv2d(fpn_dim, num_class, kernel_size=1)
		  )

		if freez_bn == True:
			self.freeze_bn()

	def forward(self, conv_out, pred_de_feats):
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
		results.append(seg_res)

		fpn_feature_list = [f]
		pred_de_feats[0] = nn.functional.upsample(pred_de_feats[0], size=f.size()[2:], mode='bilinear', align_corners=False)

		for i in reversed(range(len(conv_out) - 1)):
			conv_x = conv_out[i]
			conv_x = self.fpn_in[i](conv_x) # lateral branch

			f = crop_like(self.upscale[i](f), conv_x)
			f = conv_x + f
			pred_de_feats[2 - i] = crop_like(pred_de_feats[2-i], f)
			joint_feature = torch.cat([f, pred_de_feats[2-i]], 1)
			joint_feature = self.joint_fpn_out[i](joint_feature)

			seg_res = F.upsample(seg_res, size=conv_x.size()[2:], mode='bilinear', align_corners=False)
			seg_res = F.sigmoid(seg_res)
			joint_feature = self.att_out[i]([joint_feature, seg_res])
			seg_res = self.score_out[i](joint_feature)
			results.append(seg_res)

			fpn_feature_list.append(joint_feature)
      
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
        
class STCNN(nn.Module):
	def __init__(self, pred_enc, seg_enc, pred_dec, j_seg_dec):
		super(STCNN, self).__init__()
		self.pred_encoder = pred_enc
		self.pred_decoder = pred_dec
		self.seg_encoder = seg_enc
		self.seg_decoder = j_seg_dec

	def forward(self, seq, frame):
		pred_en_feats = self.pred_encoder(seq, return_feature_maps=True)
		pred, pred_de_feats = self.pred_decoder(pred_en_feats,return_feature_maps=True)
		pred_feats = pred_de_feats
		for i in range(len(pred_de_feats)):
			pred_feats[i] = (pred_feats[i].detach())
		seg_en_feats = self.seg_encoder(frame, return_feature_maps=True)

		seg_res = self.seg_decoder(seg_en_feats, pred_feats)

		if isinstance(seg_res,list):
			for i in range(len(seg_res)):
				seg_res[i] = F.upsample(seg_res[i], size=frame.size()[2:], mode='bilinear', align_corners=False)
		else:
			seg_res = F.upsample(seg_res, size=frame.size()[2:], mode='bilinear', align_corners=False)

		return seg_res,pred        
