import cv2
import numpy as np
from scipy.signal import convolve
import time
from commonly_used import *
import matplotlib.pyplot as plt
from time import time
import scipy.io as sio
import os
import xlrd
from multiprocessing import Pool, Process
from math import exp
from scipy.signal import convolve, convolve2d
import torch
import torch.backends.cudnn as cudnn
from Utils import Rec_Loss, Augmentation, GaussPSF

# 相机透镜模型 u是物距
def uvf(v, f):
	u = (v * f) / (v - f)
	return u


# F镜头焦距、A光圈、fd focus depth、gtd ground truth depth、N 相机的f-number、p 像素尺寸到实际尺寸的比值、s 传感单元与相机输出尺寸大小、p s is the quintessence of this formula
def getCoC(F, A, fd, gtd):
	# A=F/N
	c = abs((gtd - fd)) / gtd * F / (fd - F)
	# print(c,end=' ')
	# c=c/(p*s)
	c = A * c
	# print(c)
	return c


# 生产高斯卷积核
def getGaussianKernel(n, sigma):
	kernel = np.zeros((n, n), dtype=np.float)
	# print(kernel.shape,kernel.dtype)
	for i in range(-n // 2, n // 2):
		for j in range(-n // 2, n // 2):
			kernel[i + n // 2][j + n // 2] = (1 / (2 * np.pi * sigma * sigma)) * np.exp(
				(-1) * (i * i + j * j) / (2 * sigma))
	# print(i,j,kernel[i+n//2][j+n//2])
	kernel = kernel / np.sum(kernel)
	return kernel


def load_fd(root, f, C, c0):
	fd = np.ones(C)
	v = np.ones(C)
	print("读取传感器数据......", end=' ')
	if os.path.isfile(root + '/fd.npy') != True or read == True:
		print("从excel读取：")
		v=np.load(root+'/fd/v.npy')
		#table = xlrd.open_workbook(root + '/fd/v.xlsx').sheets()[0]  # 读取excel文件中的第一个sheet
		for i in range(C):
			#v[i] = float(table.cell(i, 0).value[1:-1])
			# print(v[i],end=' ')
			fd[i] = uvf(v[i] + c0, f)
		# print(fd[i])
		np.save(root + '/fd/fd.npy', fd)
		np.save(root + '/fd/v.npy', v)
	else:
		print("读取现有数据")
		fd = np.load(root + '/fd.npy')
	return fd


def read_img(img, d):
	dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
	dftshift = np.fft.fftshift(dft)
	rows, cols = img.shape
	crow, ccol = int(rows / 2), int(cols / 2)
	mask = np.zeros((rows, cols, 2), np.uint8)
	mask[crow - d:crow + d, ccol - d:ccol + d] = 1
	fshit = dftshift * mask
	ishit = np.fft.ifftshift(fshit)
	iImg = cv2.idft(ishit)
	iImg = cv2.magnitude(iImg[:, :, 0], iImg[:, :, 1])
	plt.subplot(121), plt.imshow(img, cmap='gray')
	plt.title('orignal'), plt.axis('off')
	plt.subplot(122), plt.imshow(iImg, cmap='gray')
	plt.title('inverse'), plt.axis('off')
	plt.show()


def loadPosition(root):
	# 此函数用于读入三张包含坐标信息的图并压入padding 1
	# 将三张图表读入
	tic = time()
	print("读入每个点的Calib XYZ")
	load_path = os.path.join(os.getcwd(), root + "/3D/X.mat")
	X_mat = np.expand_dims(np.array(sio.loadmat(load_path)["X"]), 2)
	load_path = os.path.join(os.getcwd(), root + "/3D/Y.mat")
	Y_mat = np.expand_dims(np.array(sio.loadmat(load_path)["Y"]), 2)
	load_path = os.path.join(os.getcwd(), root + "/3D/Z.mat")
	Z_mat = np.expand_dims(np.array(sio.loadmat(load_path)["Z"]), 2)
	# 将三张图标融合在一起，total形状为[row, col, 3]
	total = np.append(np.append(X_mat, Y_mat, 2), Z_mat, 2)
	toc = time()
	print("读入完成\t\t用时%f秒" % (toc - tic))
	print("--------------------------------")
	# 输出total的形状为[row, col, 3]
	return total


def calGTDepth(RootPath, TransMat):
	# 此函数用于计算Ground True Depth
	calib_Pos = loadPosition(RootPath)
	[row, col, _] = calib_Pos.shape
	# 设定Padding， 将Padding融合至calib_Pos中
	Padding = np.ones([row, col, 1])
	# 此时calib_Pos形状为[row, col, 4]
	calib_Pos = np.append(calib_Pos, Padding, 2)
	Cam_Pos = np.matmul(calib_Pos, TransMat)
	depth = np.abs(Cam_Pos[:, :, 2])
	print('true depth shape', depth.shape)
	depth = cv2.resize(depth, img_shape)
	# 获得其中所有的深度种类
	ini_depth_list = np.unique(depth)
	min_depth, max_depth = np.min(ini_depth_list), np.max(ini_depth_list)
	depth_list = np.linspace(min_depth, max_depth, 400)
	print("所有的深度在(%f, %f)之间 " % (depth_list[0], depth_list[-1]))
	print("--------------------------------")
	np.save(RootPath + '/3D/depth.npy', depth)
	np.save(RootPath + '/3D/depth_list.npy', depth_list)
	return depth, depth_list


def depth_ini(shape):
	# 此函数用于深度的初始化
	# 方形深度，从400变化至649，共250种深度
	print("检查是否存在已有深度表")
	check = os.path.isfile("depth_400_649.npy") and os.path.isfile("depth_list.npy")
	if not check:
		print("制作深度表")
		tic = time()
		num_depth = shape[0] // 2
		# 设定一个存储depth标签和数值的两列空间，便于后面切片时寻找位置
		depth_list = np.zeros(num_depth)
		depth = np.zeros(shape)
		row, col = shape[0], shape[1]
		for i in range(num_depth):
			depth[i:row - i, i:col - i].fill(400 + i)
			depth_list[i] = 400 + i
		toc = time()
		print("深度表制作完成\t用时 %f秒" % (toc - tic))
		print("--------------------------------")
		np.save("depth_400_649.npy", depth)
		np.save("depth_list.npy", depth_list)

	else:
		print("读入深度表")
		print("--------------------------------")
		depth = np.load("depth_400_649.npy")
		depth_list = np.load("depth_list.npy")
	return depth, depth_list


def PSF(img, depth, F, A, ksize, fd):
	# 检查输入图像通道数 目前只支持单通道
	# 检查图像和深度图的形状是否匹配
	simu_img = np.ones((img.shape))
	for i in range(0, img.shape[0]):
		print(i)
		for j in range(0, img.shape[1]):
			c = getCoC(F=F, A=A, fd=fd, gtd=depth[i][j])
			if c != 0:
				kernel = getGaussianKernel(ksize, 500 * c)
				simu_img[i][j] = point_convolve(img, kernel, i, j)
			else:
				simu_img[i][j] = img[i][j]

	simu_img = simu_img.astype("uint8")
	# cv2.imshow("test",simu_img)
	# cv2.imwrite("psf"+str(fd)+".png",simu_img)
	# edge=cv2.Laplacian(simu_img,cv2.CV_8UC1,ksize=7)
	# cv2.imshow("edge",edge)

	return simu_img


def PSF_silce(img, depth, depth_list, F, A, w, fd, dd, id):
	num_depth = depth_list.shape[0]
	img_list = np.zeros(shape=[num_depth, img.shape[0], img.shape[1]])
	simu_img = np.ones(img.shape)
	# 把每一种depth带来的卷积操作执行完成
	print(id, end=',')
	# print("针对每一种深度，执行卷积操作")
	tic = time()
	for i in range(num_depth):
		# 针对depth_list种的每一个深度，寻找一次c，并做一次卷积操作
		c = getCoC(F=F, A=A, fd=fd, gtd=depth_list[i])
		# ksize=int(c)//2*2+1
		# print('A:',A,' C:',c)
		ksize = int((w * c) // 2) * 2 + 1
		img_list[i, :, :] = cv2.GaussianBlur(img, (ksize, ksize), c / 2)
	# ksize=int((w*c)//2)*2+1
	# img_list[i, :, :] = cv2.GaussianBlur(img, (w, w),c/2)
	# 开始挑选点
	toc = time()
	# print("卷积操作完成\t用时%f秒"%(toc-tic))
	# print("--------------------------------")
	row, col = img.shape
	# # 由于形状的特殊性，这里采用fill的方式填充
	# for i in range(row//2):
	#     simu_img[i:row-i, i:row-i] = img_list[i, i:row-i, i:row-i]

	# 对于一般情况, 采用以下这种方式
	# print("按照每个点的深度，依次寻找对应卷积值")
	tic = time()
	for i in range(row):
		for j in range(col):
			# 首先是读入这个点的深度 depth[i, j]
			cur_depth = depth[i, j]
			# 寻找index，并修改此处图像点
			index = np.argwhere(depth_list >= cur_depth)[0][0]
			simu_img[i, j] = img_list[index, i, j]
	toc = time()
	# print("查询完成\t\t用时%f秒"%(toc-tic))
	# print("--------------------------------")
	simu_img = simu_img.astype("uint8")
	# print(root+"/simu"+str(id)+".bmp")
	if not os.path.exists(os.path.join(root, "temp2")):
		# 判断文件夹是否存在，不存在则创建文件夹
		os.mkdir(os.path.join(root, "temp2"))
	cv2.imwrite(root + "/temp2/simu" + str(A) + '_' + str(w) + '_' + str(dd) + ".bmp", simu_img)
	return simu_img


def generate_simu_img(clear_img, depth, depth_list, fd, F, rangeA, rangew, ranged, num_pool=12):
	num = 0
	p = Pool(num_pool)
	for ww in rangew:
		for aa in rangeA:
			for dd in ranged:
				# 生成3张图片
				# print(aa,ww,dd)
				# print(num)
				p.apply_async(PSF_silce, args=(clear_img, depth, depth_list, F, aa, ww, fd[dd], dd, num))
				num += 1
	p.close()
	p.join()


# 输入全为CPU下numpy格式的数据
def generate_simu_img_cuda(clear_img, depth, fds, F, rangeA, rangew, ranged, theta):

	# ························第2种方式························#
	R = clear_img[:, :, 2]
	G = clear_img[:, :, 1]
	B = clear_img[:, :, 0]
	H = cv2.cvtColor(clear_img, cv2.COLOR_BGR2GRAY)
	weights_hsv = theta * 1 / (R / 700 + G / 546.1 + B / 435.8) / 180.038470
	weights_hsv = cv2.normalize(weights_hsv, None, alpha=0.8, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
	# weights_hsv=np.ones(clear_img.shape)
	weights_hsv[:]=1
	print(weights_hsv)
	# cv2.imshow('hsl_imagev',weights_hsv)
	# cv2.waitKey(0)
	weights_hsv_temp = torch.Tensor(weights_hsv).float().cuda()
	# weights_hsv=hsl_image

	# ························第3种方式························#
	# B435.8 0.777，G5460.975   R700 1.248
	weights_rgb = np.array([1, 1, 1])
	weights_rgb = torch.Tensor(weights_rgb).float().cuda()

	# 清晰图归一化处理
	clear_img = cv2.normalize(clear_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
	clear_img = np.transpose(clear_img, (2, 0, 1))
	# print(clear_img.shape)
	image = torch.torch.zeros(1, 3, clear_img.shape[1], clear_img.shape[2])
	gt_depth = torch.torch.zeros(1, clear_img.shape[1], clear_img.shape[2])
	print(gt_depth.shape)
	print(image.shape)
	print(depth.shape)
	print(clear_img.shape)
	gt_depth[0, :, :] = torch.torch.from_numpy(depth)

	weights_hsv = torch.torch.zeros(1, img_shape[1], img_shape[0])
	print("weights_hsv.shape",weights_hsv.shape)
	weights_hsv[0, :, :] = weights_hsv_temp

	image[0] = torch.torch.from_numpy(clear_img)
	focal_length = torch.Tensor([F]).float().cuda()
	for w in range(len(rangew)):
		for ape in range(len(rangeA)):
			print(f'ape={rangeA[ape]}', end=',')
			for fd in range(len(ranged)):
				focal_depth = torch.Tensor([fds[ranged[fd]]]).float().cuda()
				aperture = torch.Tensor([rangeA[ape]]).float().cuda()
				# print(focal_depth)

				focused = generator(image, gt_depth, focal_depth, aperture, 5, focal_length, weights_hsv, weights_rgb)

				focused1 = (focused[0, :, :].permute(1, 2, 0).cpu().detach().numpy() * 255).astype('uint8')
				#focused1=np.rot90(np.flip(focused1,axis=1),1)
				cv2.imshow('focused1', focused1)
				cv2.waitKey(20)
				if not os.path.exists(os.path.join(root, "temp2")):
					os.mkdir(os.path.join(root, "temp2"))
				cv2.imwrite(
					root + "/temp2/simu" + str(rangeA[ape]) + '_' + str(theta) + '_' + str(ranged[fd]) + ".bmp",
					focused1)
		# print(ape,fd)
	cv2.destroyAllWindows()
	print('')


def load_gt_img(ranged):
	# print(len(ranged))
	N = len(ranged)
	gt_img = np.zeros((img_shape[1], img_shape[0], N), dtype='uint8')
	for i in range(N):
		temp = cv2.imread(root + '/' + img_name + '/' + img_name + '_' + str(ranged[i] + 1) + '.bmp', 0)

		gt_img[:, :, i] = cv2.resize(temp, img_shape)
		#cv2.imshow('test',gt_img[:, :, i] )
		#cv2.waitKey(0)
	return gt_img


def comp_loss(gt_img, rangeA, rangew, ranged, num_pool=5):
	loss = np.zeros((len(rangeA), 1))
	# np.save(root+'/temp_loss.npy',loss)
	# print(loss.shape)
	results = []
	p = Pool(num_pool)
	for a in range(len(rangeA)):
		#for w in range(len(rangew)):
			# print('a=',a,'w=',w)
		results.append(p.apply_async(loss_all, args=(root, gt_img, rangeA[a],rangew , ranged)))
	p.close()
	p.join()
	# print(results)
	num = 0
	for a in range(len(rangeA)):
		#for w in range(len(rangew)):
		loss[a, 0] = results[num].get()
		num += 1
	# print(loss[a,w])
	#print(loss)
	#print(loss.shape)
	##print(type(loss))
	return loss


# loss 多线程调用
def loss_all(root, gt_img, a, w, ranged):
	loss0, loss1, loss2, loss3 = 0, 0, 0, 0
	# loss=np.load(root+'/temp_loss.npy')
	# print('a=',a,' w=',w)
	for dd in range(len(ranged)):
		#print('a=',a,' w=',w,'d=',ranged[dd],end=' ')
		simu_img = cv2.imread(root + "/temp2/simu" + str(a) + '_' + str(w) + '_' + str(ranged[dd]) + ".bmp", 0)
		true_blur_img = gt_img[:, :, dd]
		#cv2.imshow('true'+str(a),true_blur_img)
		#cv2.imshow('simu'+str(a),simu_img)
		#cv2.waitKey(0)
		# print(simu_img.shape,true_blur_img.shape)
		# print(true_blur_img)
		loss0 += L2_loss(simu_img, true_blur_img) / (img_shape[0] * img_shape[1]) * 50000
		loss1 += defocus_loss(simu_img, true_blur_img)
		loss2 += defocus_num_loss(simu_img, true_blur_img) * 4
		loss3 += calRecError(simu_img, true_blur_img) * 5000
	print(loss0, loss1, loss2, loss3)
	# loss=(loss0+loss1+loss2)/len(ranged)
	loss = (loss0 + loss1 + loss2 + loss3) / (len(ranged) * 4)
	# loss=(loss3)/(len(ranged))

	print('a=', a, ' w=', w, ' loss=', loss)
	# np.save(root+'/temp_loss.npy',loss)
	return loss


# LOSS 之间计算F范数
def L2_loss(img1, img2):
	return (np.linalg.norm(img1.astype('float').flatten() - img2.astype('float').flatten(), ord=2))


# 对图像求一次二阶导然后求F范数
def defocus_loss(img1, img2):
	edge1 = cv2.Laplacian(img1, cv2.CV_64F, ksize=5)
	edge1 = cv2.convertScaleAbs(edge1)
	edge2 = cv2.Laplacian(img2, cv2.CV_64F, ksize=5)
	edge2 = cv2.convertScaleAbs(edge2)
	loss = np.linalg.norm(edge1.astype('float').flatten() - edge2.astype('float').flatten(), ord=2) / (
				img1.shape[0] * img1.shape[1] / 10000)
	return loss


# 统计每个清晰度的个数 并求loss 后期有空加入格式输入检查 优化算法
def defocus_num_loss(img1, img2):
	tic = time()
	# 拉普拉斯 图像清晰度可以看作是二阶导
	edge1 = cv2.Laplacian(img1, cv2.CV_64F, ksize=5)
	edge2 = cv2.Laplacian(img2, cv2.CV_64F, ksize=5)

	# 把Laplace得到的数据也就是defocus数据分为1000项
	classes = 1000
	minab = np.min((np.min(edge1), np.min(edge2)))
	maxab = np.max((np.max(edge1), np.max(edge2)))

	defocus_num_list = np.linspace(minab, maxab, classes + 1)
	# print(defocus_num_list,defocus_num_list.shape)
	# 统计每个清晰度区间的个数 很慢 不知道有没有更快的操作
	num1 = np.zeros(classes)
	num2 = np.zeros(classes)
	for i in range(classes):
		num1[i] = ((edge1 >= defocus_num_list[i]) & (edge1 <= defocus_num_list[i + 1])).sum()
		num2[i] = ((edge2 >= defocus_num_list[i]) & (edge2 <= defocus_num_list[i + 1])).sum()
	# print(num1)
	# print(num2)
	toc = time()
	# 归一化 取消图像大小对loss影响
	loss = L2_loss(num1, num2) / (img1.shape[0] * img1.shape[1] / 10000)
	# print(loss,(toc-tic),'s')
	return loss


def create_gaussian(window_size, sigma):
	# 此函数用于经计算SSIM中的高斯核
	gauss = np.array([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
	gauss = gauss / np.sum(gauss)
	window_1 = np.expand_dims(gauss, 1)
	window = np.matmul(window_1, window_1.transpose())
	return window


def calSSIM(img1, img2, window_size, sigma=1.5):
	# 此函数用于计算SSIM
	window = create_gaussian(window_size, sigma)
	# 分别求公式中的mu 和 sigma
	mu1 = convolve2d(img1, window, "same")
	mu2 = convolve2d(img2, window, "same")
	mu1_sq = np.power(mu1, 2)
	mu2_sq = np.power(mu2, 2)
	mu1_mu2 = mu1 * mu2

	sigma1_sq = convolve2d(img1 * img1, window, "same") - mu1_sq
	sigma2_sq = convolve2d(img2 * img2, window, "same") - mu2_sq
	sigma12 = convolve2d(img1 * img2, window, "same") - mu1_mu2

	C1 = 0.01 ** 2
	C2 = 0.03 ** 2
	ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
	return ssim


def calRecError(img1, img2, window_size=11, sigma=1.5):
	alpha = 1
	img1 = img1.astype('float')
	img2 = img2.astype('float')
	ssim_loss = np.mean(1 - calSSIM(img1, img2, window_size, sigma=1.5)) / 2
	rec_loss = np.linalg.norm(img1.flatten() - img2.flatten(), ord=1)
	return alpha * ssim_loss + (1 - alpha) * rec_loss


roots = [
	'F:/OneDrive - stu.cqupt.edu.cn/code/SFF_code/data/9.27true1',
	'F:/OneDrive - stu.cqupt.edu.cn/code/SFF_code/data/9.27true2',
	'F:/OneDrive - stu.cqupt.edu.cn/code/SFF_code/data/9.27true3',
	'F:/OneDrive - stu.cqupt.edu.cn/code/SFF_code/data/9.27true4',
	'F:/OneDrive - stu.cqupt.edu.cn/code/SFF_code/data/9.27true5',

	'F:/OneDrive - stu.cqupt.edu.cn/code/SFF_code/data/9.25boox1',
	'F:/OneDrive - stu.cqupt.edu.cn/code/SFF_code/data/9.25boox2',

	'F:/OneDrive - stu.cqupt.edu.cn/code/SFF_code/data/9.23mask1',
	'F:/OneDrive - stu.cqupt.edu.cn/code/SFF_code/data/9.23mask2',
	'F:/OneDrive - stu.cqupt.edu.cn/code/SFF_code/data/9.23mask3',
	'F:/OneDrive - stu.cqupt.edu.cn/code/SFF_code/data/9.23mask4',
	'F:/OneDrive - stu.cqupt.edu.cn/code/SFF_code/data/9.23mask5',
	'F:/OneDrive - stu.cqupt.edu.cn/code/SFF_code/data/9.27true6',

	'F:/OneDrive - stu.cqupt.edu.cn/code/SFF_code/data/9.07mask1',
	'F:/OneDrive - stu.cqupt.edu.cn/code/SFF_code/data/9.07mask2',
	'F:/OneDrive - stu.cqupt.edu.cn/code/SFF_code/data/9.07mask3'


]

#np_old_load=np.load

# img_name=root[-5:]
# print(img_name)
# img_shape = (500, 500)
# root='F:/OneDrive - stu.cqupt.edu.cn/code/SFF_code/data/7.24mask1'
# root='I:/heka/SFF/PSF/data/7.24mask1'
# img_name='mask1'
img_shape = ( 613,514)
#img_shape = ( 514,613)
#
if __name__ == '__main__':
	

	for i in range(len(roots)):
		global root, img_name
		root = roots[i]
		img_name = root[-5:]
		# ············cuda初始化············#
		cudnn.benchmark = True
		generator = GaussPSF(7, near=1e-5, far=1e7)
		generator = torch.nn.DataParallel(generator)
		generator = generator.cuda()
		torch.set_num_threads(1)


		# ···························································读入清晰图片 clear image···························································#
		print('读入清晰图片')
		tic = time()
		clear_img = cv2.imread(root + "/clear.tif")
		print(root)
		print(clear_img.shape)
		clear_img = cv2.resize(clear_img, img_shape)
		toc = time()
		print(clear_img.shape)
		print(f'读入完成    耗时{(toc - tic):.2f}s')
		print("--------------------------------")
		# ···························································读入深度 ground truth depth···························································#

		if os.path.isfile(root + '/3D/depth.npy') != True :
			depth, depth_list = calGTDepth(root, Trans_mat0907)
			print(depth.shape)
		else:
			print("读取现有数据")
			depth = np.load(root + '/3D/depth.npy',allow_pickle=True)
			depth_list = np.load(root + '/3D/depth_list.npy',allow_pickle=True)
			depth=cv2.resize(depth,img_shape)
			print("读取完毕!")
			print("--------------------------------")



		#depth=np.flip(depth,axis=1)
		#cv2.imshow('test', depth.astype('uint8'))
		#cv2.waitKey(0)
		#fig = show_3D_picture(depth)
		#depth=depth.T

		depth=np.flip(depth,axis=1)
		#depth=np.flip(depth,axis=0)
		#depth = np.rot90(depth, 2)

		#fig=show_3D_picture(depth[000:500,000:500])
		#depth = depth.T
		print('depthshape',depth.shape)
		#depth=np.rot90(depth,-1)
		depth = np.ascontiguousarray(depth)
		print('depthshape',depth.shape)

		cv2.imshow("test_depth",cv2.normalize(depth, None, alpha=1, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1))
		cv2.imshow('clear',clear_img)
		cv2.waitKey(10)
		#cv2.destroyAllWindows()
		#fig=show_3D_picture(depth)
		
		#fig=show_3D_picture(depth)
		# depth2=np.flip(depth2,axis=0)
		# fig=show_3D_picture(depth2)
		# ···························································读入焦深 focus depth···························································#
		read = 1  # 1 强制重新读取 0 读入现有数据
		F = 50
		C = 200
		c0 = 24.5
		fd = load_fd(root, F, C, c0)
		print("读取完毕!")
		print(f"所有的focus depth在({fd[0]}, {fd[-1]})之间")
		print("--------------------------------")

		# ··························································暴力搜索A w范围···························································#
		F = 50  # 设置镜头参数
		# 设置求解参数的范围 A为光圈参数A/(p*s) w为卷积核大小 使用高斯函数时只能为单数 d为选择的照片序列
		rangeA = np.array(range(400, 1200, 50))
		rangew = np.array(range(4, 5, 1))
		# ranged=np.array(range(0,200,10))
		ranged = np.array(range(0, 200, 10))
		rangec = np.array(range(0, 10, 1))

		# 初始化loss矩阵
		loss = np.zeros((len(rangeA), len(rangec)))
		print(loss.shape)
		tic = time()

		loss_c = np.zeros(len(rangec))
		# 可以对每个w计算一次loss 也可以全部把图片求出来之后再计算loss 前一种方法实时性较好
		for cc in range(len(rangec)):
			# for i in range(len(rangew)):
			c0=23+cc*0.2
			#c0 = 24.2
			fds = load_fd(root, F, C, c0)
			print(f"c0为{c0},所有的focus depth在({fds[0]}, {fds[-1]})之间")
			# 生成模拟模糊图片
			#generate_simu_img(clear_img,depth,depth_list,fds,F,rangeA,rangew,ranged,num_pool=3)
			theta = 1 + cc
			print(f'theta:{theta}')
			#generate_simu_img_cuda(clear_img, depth, fds, F, rangeA, rangew, ranged, theta)
			# 载入真实模拟图片
			gt_img = load_gt_img(ranged)
			# 计算loss
			# temploss= comp_loss(gt_img,rangeA,rangew[i:i+1],ranged)
			temploss = comp_loss(gt_img, rangeA, theta, ranged, num_pool=5)
			# 通过第一种方法计算出来只是一条loss 需要拼接一次
			# loss [:,i]=temploss[:,0]
			# np.save(root+'/preloss.npy',loss)
			#print(type(temploss))
			#print(temploss.shape)
			loss[:, cc] = temploss[:,0]
			np.save(root + '/preloss.npy', loss)
			print('当前最小值为', np.min(loss[:, 0:cc + 1]), '位于', np.where(loss == np.min(loss[:, 0:cc + 1])))

		np.save(root + '/preloss.npy', loss)
		print(np.min(loss), np.where(loss == np.min(loss)))
