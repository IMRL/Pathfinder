import os
import yaml
import numpy as np
from PIL import Image
import cv2


class LivoxPrediction(object):
    def __init__(self, root, sequences):
        self.root = root
        self.sequences = sequences
        self.sequences.sort()  # sort seq id

        if os.path.isdir(self.root):
            print("Dataset found: {}".format(self.root))
        else:
            raise ValueError("dataset not found: {}".format(self.root))

        self.label_files = []
        self.imgmask_files = []

        for seq in self.sequences:
            # format seq id
            seq = "{0:02d}".format(int(seq))
            print("parsing seq {}...".format(seq))

            label_path = os.path.join(self.root, seq, "label")
            label_files = [os.path.join(label_path, f)
                           for f in os.listdir(label_path) if ".label" in f]

            self.label_files.extend(label_files)

            imgmask_path = os.path.join(self.root, seq, "imgmask")
            imgmask_files = [os.path.join(imgmask_path, f)
                          for f in os.listdir(imgmask_path) if ".png" in f]
            self.imgmask_files.extend(imgmask_files)

        self.imgmask_files.sort()
        self.label_files.sort()

        print("Using {} pointclouds predictions from sequences {}".format(
            len(self.label_files), self.sequences))
        print("Using {} image masks from sequences {}".format(
            len(self.imgmask_files), self.sequences))

    def loadLabelByIndex(self, index):
        sem_label = self.readLabel(self.label_files[index])
        return np.where(sem_label != 0, 100, 70)

    def loadMaskByIndex(self, index):
        imgmask = self.readMask(self.imgmask_files[index])
        return imgmask

    @staticmethod
    def readLabel(path):
        label = np.fromfile(path, dtype=np.int32)
        sem_label = label
        return sem_label

    @staticmethod
    def readImgmask(path):
        imgmask = Image.open(path)
        imgmask = np.array(imgmask)
        return imgmask

    def __len__(self):
        return len(self.label_files)


class Livox(object):
    def __init__(self, root,  # directory where data is
                 sequences,  # sequences for this data (e.g. [1,3,4,6])
                 config_path,  # directory of config file
                 has_image=True,
                 has_pcd=True,
                 has_imgmask=True,
                 has_label=True):
        # global label_files, image_files, mask_files
        self.root = root
        self.sequences = sequences
        self.sequences.sort()  # sort seq id
        self.has_label = has_label
        self.has_image = has_image
        self.has_pcd = has_pcd
        self.has_imgmask = has_imgmask

        # check file exists
        if os.path.isfile(config_path):
            self.data_config = yaml.safe_load(open(config_path, "r"))
        else:
            raise ValueError("config file not found: {}".format(config_path))

        if os.path.isdir(self.root):
            print("Dataset found: {}".format(self.root))
        else:
            raise ValueError("dataset not found: {}".format(self.root))

        self.pointcloud_files = []
        self.label_files = []
        self.image_files = []
        self.imgmask_files = []
        self.proj_matrix = {}

        for seq in self.sequences:
            # format seq id
            seq = "{0:02d}".format(int(seq))
            print("parsing seq {}...".format(seq))

            if self.has_pcd:
            # get file list from path
                pointcloud_path = os.path.join(self.root, seq, "livox")
                pointcloud_files = [os.path.join(pointcloud_path, f) for f in os.listdir(
                    pointcloud_path) if ".bin" in f]

            if self.has_label:
                label_path = os.path.join(self.root, seq, "label")
                label_files = [os.path.join(label_path, f)
                               for f in os.listdir(label_path) if ".label" in f]
            if self.has_image:
                image_path = os.path.join(self.root, seq, "image")
                image_files = [os.path.join(image_path, f) for f in os.listdir(
                    image_path) if ".jpg" in f]

            if self.has_imgmask:
                imgmask_path = os.path.join(self.root, seq, "imgmask")
                imgmask_files = [os.path.join(imgmask_path, f) for f in os.listdir(
                    imgmask_path) if ".png" in f]

            if self.has_pcd:
                if self.has_label:
                    assert (len(pointcloud_files) == len(label_files))
                if self.has_image:
                    assert (len(pointcloud_files) == len(image_files))
                if self.has_imgmask:
                    assert (len(pointcloud_files) == len(imgmask_files))

            self.pointcloud_files.extend(pointcloud_files)
            if self.has_label:
                self.label_files.extend(label_files)
            if self.has_image:
                self.image_files.extend(image_files)
            if self.has_imgmask:
                self.imgmask_files.extend(imgmask_files)

            # load calibration file
            if self.has_image and self.has_pcd:
                calib_path = os.path.join(self.root, seq, "calib.txt")
                calib = self.read_calib(calib_path)
                proj_matrix = {
                    "Extrinsic": calib["Extrinsic"],
                    "Intrinsic": calib["Intrinsic"],
                    "Distortion": calib["Distortion"]
                }
                self.proj_matrix = proj_matrix

        # sort for correspondance 为了匹配而进行排序
        if self.has_pcd:
            self.pointcloud_files.sort()
        if self.has_label:
            self.label_files.sort()
        if self.has_image:
            self.image_files.sort()
        if self.has_imgmask:
            self.imgmask_files.sort()
        print("Using {} pointclouds from sequences {}".format(
            len(self.pointcloud_files), self.sequences))

        # load config -------------------------------------
        # get color map
        sem_color_map = self.data_config["color_map"]
        max_sem_key = 0
        for k, v in sem_color_map.items():
            if k + 1 > max_sem_key:
                max_sem_key = k + 1
        self.sem_color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32)
        for k, v in sem_color_map.items():
            self.sem_color_lut[k] = np.array(v, np.float32) / 255.0

        sem_color_inv_map = self.data_config["color_map_inv"]
        max_sem_key = 0
        for k, v in sem_color_inv_map.items():
            if k + 1 > max_sem_key:
                max_sem_key = k + 1
        self.sem_color_lut_inv = np.zeros((max_sem_key + 100, 3), dtype=np.float32)
        for k, v in sem_color_inv_map.items():
            self.sem_color_lut_inv[k] = np.array(v, np.float32) / 255.0

        self.inst_color_map = np.random.uniform(
            low=0.0, high=1.0, size=(10000, 3))

        # get learning class map
        # map unused classes to used classes
        learning_map = self.data_config["learning_map"]
        max_key = 0
        for k, v in learning_map.items():
            if k > max_key:
                max_key = k
        # +100 hack making lut bigger just in case there are unknown labels
        self.class_map_lut = np.zeros((max_key + 100), dtype=np.int32)
        for k, v in learning_map.items():
            self.class_map_lut[k] = v
        # learning map inv
        learning_map = self.data_config["learning_map_inv"]
        max_key = 0
        for k, v in learning_map.items():
            if k > max_key:
                max_key = k
        # +100 hack making lut bigger just in case there are unknown labels
        self.class_map_lut_inv = np.zeros((max_key + 100), dtype=np.int32)
        for k, v in learning_map.items():
            self.class_map_lut_inv[k] = v

        # compute ignore class by content ratio
        cls_content = self.data_config["content"]
        content = np.zeros(len(self.data_config["learning_map_inv"]), dtype=np.float32)
        for cl, freq in cls_content.items():
            x_cl = self.class_map_lut[cl]
            content[x_cl] += freq
        self.cls_freq = content

        self.mapped_cls_name = self.data_config["mapped_class_name"]

    @staticmethod
    def read_calib(calib_path):
        """
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, 'r') as f:
            for line in f.readlines():
                if line == '\n':
                    break
                key, value = line.split(':', 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {'Extrinsic': calib_all['Extrinsic'].reshape(4, 4),
                     'Intrinsic': calib_all['Intrinsic'].reshape(3, 3),
                     'Distortion': calib_all['Distortion'].reshape(-1)}
        # 3x4 projection matrix for Extrinsic, 3x3 rectifying rotation matrix for Intrinsic,
        # 1x5 distortion matrix for Distortion
        return calib_out

    @staticmethod
    def readPCD(path):
        pcd = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        return pcd

    @staticmethod
    def readLabel(path):
        label = np.fromfile(path, dtype=np.int32)
        sem_label = label
        return sem_label

    @staticmethod
    def readImgmask(path):
        imgmask = Image.open(path)
        imgmask = np.array(imgmask)
        return imgmask

    def parsePathInfoByIndex(self, index):
        path = self.pointcloud_files[index]
        # linux path
        if "\\" in path:
            # windows path
            path_split = path.split("\\")
        else:
            path_split = path.split("/")
        seq_id = path_split[-3]
        frame_id = path_split[-1].split(".")[0]
        return seq_id, frame_id

    def labelMapping(self, label):
        label = self.class_map_lut[label]
        return label

    def loadLabelByIndex(self, index):
        sem_label = self.readLabel(self.label_files[index])
        return sem_label

    def loadDataByIndex(self, index):
        pointcloud = self.readPCD(self.pointcloud_files[index])
        if self.has_label:
            sem_label = self.readLabel(self.label_files[index])
        else:
            sem_label = np.zeros(pointcloud.shape[0], dtype=np.int32)
        return pointcloud, sem_label

    def loadImage(self, index):
        return Image.open(self.image_files[index])

    def loadImgmask(self, index):
        return Image.open(self.imgmask_files[index])

    def mapLidar2Camera(self, seq, pointcloud, img_h, img_w):
        if not self.has_image:
            raise ValueError("cannot mappint pointcloud with has_image=False")

        proj_matrx = self.proj_matrix

        pointcloud_hcoord = np.hstack((pointcloud[:, :3], np.ones((pointcloud.shape[0], 1))))
        transformed_points = pointcloud_hcoord.dot(proj_matrx['Extrinsic'].T)[:, :3]
        points_2d, _ = cv2.projectPoints(transformed_points.reshape(-1, 1, 3),
                                         np.zeros(3), np.zeros(3),
                                         proj_matrx['Intrinsic'], proj_matrx['Distortion'])
        mapped_points = points_2d.reshape(-1, 2)

        return mapped_points

    def __len__(self):
        return len(self.pointcloud_files)


class LivoxRGB(Livox):
    def __init__(self, root,  # directory where data is
                 sequences,  # sequences for this data (e.g. [1,3,4,6])
                 config_path,  # directory of config file
                 has_image=True,
                 has_pcd=True,
                 has_mask=True,
                 has_label=True):
        self.root = root
        self.sequences = sequences
        self.sequences.sort()  # sort seq id
        self.has_label = has_label
        self.has_image = has_image
        self.has_pcd = has_pcd
        self.has_mask = has_mask

        # check file exists
        if os.path.isfile(config_path):
            self.data_config = yaml.safe_load(open(config_path, "r"))
        else:
            raise ValueError("config file not found: {}".format(config_path))

        if os.path.isdir(self.root):
            print("Dataset found: {}".format(self.root))
        else:
            raise ValueError("dataset not found: {}".format(self.root))

        self.pointcloud_files = []
        self.label_files = []
        self.image_files = []
        self.imgmask_files = []
        self.proj_matrix = {}

        for seq in self.sequences:
            # format seq id
            seq = "{0:02d}".format(int(seq))
            print("parsing LivoxRGB seq {}...".format(seq))

            # get file list from path
            pointcloud_path = os.path.join(self.root, seq, "eight_channel_data")  # xyzidrgb
            pointcloud_files = [os.path.join(pointcloud_path, f) for f in os.listdir(
                pointcloud_path) if ".bin" in f]

            label_path = os.path.join(self.root, seq, "label")
            label_files = [os.path.join(label_path, f)
                           for f in os.listdir(label_path) if ".label" in f]

            image_path = os.path.join(self.root, seq, "image")
            image_files = [os.path.join(image_path, f) for f in os.listdir(
                image_path) if ".jpg" in f]

            imgmask_path = os.path.join(self.root, seq, "imgmask")
            imgmask_files = [os.path.join(imgmask_path, f) for f in os.listdir(
                imgmask_path) if ".png" in f]

            if self.has_pcd:
                if self.has_label:
                    assert (len(pointcloud_files) == len(label_files))
                if self.has_image:
                    assert (len(pointcloud_files) == len(image_files))
                if self.has_mask:
                    assert (len(pointcloud_files) == len(imgmask_files))

            self.pointcloud_files.extend(pointcloud_files)
            self.label_files.extend(label_files)
            self.image_files.extend(image_files)
            self.imgmask_files.extend(imgmask_files)

            # load calibration file
            if self.has_image:
                calib_path = os.path.join(self.root, seq, "calib.txt")
                calib = self.read_calib(calib_path)
                proj_matrix = {
                    "Extrinsic": calib["Extrinsic"],
                    "Intrinsic": calib["Intrinsic"],
                    "Distortion": calib["Distortion"]
                }
                self.proj_matrix = proj_matrix

        # sort for correspondance
        if self.has_pcd:
            self.pointcloud_files.sort()
        if self.has_label:
            self.label_files.sort()
        if self.has_image:
            self.image_files.sort()
        if self.has_imgmask:
            self.imgmask_files.sort()
        print("Using {} LivoxRGB pointclouds from sequences {}".format(
            len(self.pointcloud_files), self.sequences))

        # load config -------------------------------------
        # get color map
        sem_color_map = self.data_config["color_map"]
        max_sem_key = 0
        for k, v in sem_color_map.items():
            if k + 1 > max_sem_key:
                max_sem_key = k + 1
        self.sem_color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32) # 初始化语义颜色查找表
        for k, v in sem_color_map.items():
            self.sem_color_lut[k] = np.array(v, np.float32) / 255.0

    @staticmethod
    def readPCD(path):
        pcd = np.fromfile(path, dtype=np.float32).reshape(-1, 8)
        return pcd