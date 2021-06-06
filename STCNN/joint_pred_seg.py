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
