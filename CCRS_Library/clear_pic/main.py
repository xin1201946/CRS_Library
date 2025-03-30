import matplotlib.pyplot as plt
import math
import random
import numpy as np
import os
from math import sin, cos, radians, fabs
import cv2
import importlib.resources

# 初始化模型为 None，后续会加载具体的模型
# Initialize the model to None, and load the specific model later
model = None
# 获取 'tab10' 颜色映射的前 10 种颜色
# Get the first 10 colors from the 'tab10' color map
colors = plt.cm.get_cmap('tab10')(range(10))
# 将颜色值从 [0, 1] 范围转换为 [0, 255] 范围，并转换为整数类型
# Convert the color values from the range [0, 1] to [0, 255] and convert to integer type
colors = (colors * 255).astype(int)

def main(img, save_Path=None, filename=None):
    """
    进行处理图片
    Process the image.

    :param img: 二值化后的图像
    :param img: The binarized image.
    :param save_Path: 保存位置
    :param save_Path: The save path.
    :return: 经过裁剪的图像
    :return: The cropped image.
    """
    if model is None:
        # 如果模型未加载，则加载模型
        # If the model is not loaded, load the model
        load_model()
    else:
        pass
    # 处理图像文件
    # Process the image file
    imgs = process_image_file(img, save_Path, filename)
    return imgs

def load_model():
    """
    加载 YOLOv5 模型。
    Load the YOLOv5 model.

    Returns:
        YOLOv5: 加载好的 YOLOv5 模型。
        YOLOv5: The loaded YOLOv5 model.
    """
    from yolov5 import YOLOv5
    global model
    # 使用 importlib.resources 获取模型文件的路径
    # Use importlib.resources to get the path of the model file
    with importlib.resources.path("CCRS_Library.clear_pic", "NumVision.pt") as model_path:
        # 加载模型，假设 YOLOv5 接受字符串类型的路径
        # Load the model, assuming YOLOv5 accepts a string-type path
        model = YOLOv5(str(model_path))
        # 设置置信度阈值
        # Set the confidence threshold
        model.conf = 0.6
    return model

def crop_and_save(img, detections, save_dir='./cache/', filename='crop'):
    """
    将图像裁剪到模型识别到的区域，并保存到指定文件夹
    Crop the image to the region recognized by the model and save it to the specified folder.

    Args:
        img: 原图
        img: The original image.
        detections: 模型检测结果，列表类型，每个元素为 [x1, y1, x2, y2, confidence, class]
        detections: The model detection results, in list type, each element is [x1, y1, x2, y2, confidence, class].
        save_dir: 保存裁剪图像的文件夹路径
        save_dir: The folder path to save the cropped image.
    """
    # 创建保存文件夹
    # Create the save folder
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cropped_img = img
    # 如果检测结果为空，则保存原图
    # If the detection results are empty, save the original image
    if not detections:
        cv2.imwrite(os.path.join(save_dir, f'{filename}.jpg'), img)
        return img

    # 裁剪并保存检测区域
    # Crop and save the detected regions
    for i, detection in enumerate(detections):
        x1, y1, x2, y2, _, _ = detection
        cropped_img = img[int(y1):int(y2), int(x1):int(x2)]
        # 保存裁剪图像
        # Save the cropped image
        cv2.imwrite(os.path.join(save_dir, f'{filename}.jpg'), cropped_img)
    return cropped_img

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    """
    Draws a box on an image with optional label
    在图像上绘制一个带可选标签的框。

    Args:
        x: Bounding box coordinates in (top, left, bottom, right) format
        x: 边界框坐标，格式为 (top, left, bottom, right)
        img: Image to draw on
        img: 要绘制的图像
        color: Color to draw box
        color: 框的颜色
        label: Optional label to display
        label: 可选的显示标签
        line_thickness: Thickness of the box lines
        line_thickness: 框线的厚度
    """
    # 计算线/字体的厚度
    # Calculate the line/font thickness
    tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    if color is None:
        # 如果未提供颜色，则生成随机颜色
        # Generate a random color if not provided
        color = [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # 绘制矩形框
    # Draw a rectangle
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        # 如果有标签，则绘制标签
        # If there is a label, draw the label
        tf = max(tl - 1, 1)  # 字体厚度
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # 填充矩形
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def correct_orientation(cropped_img):
    """
    校正裁剪图像的方向。
    Correct the orientation of the cropped image.

    Args:
        cropped_img: 裁剪后的图像
        cropped_img: The cropped image.

    Returns:
        numpy.ndarray: 校正后的图像
        numpy.ndarray: The corrected image.
    """
    # 转换为灰度图
    # Convert to grayscale
    gray_image = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

    # 使用自适应阈值增强边缘检测效果
    # Use adaptive thresholding to enhance edge detection
    thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # 霍夫直线变换
    # Hough line transform
    lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)

    if lines is not None and len(lines) > 0:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 计算直线的角度
            # Calculate the angle of the line
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angles.append(angle)

        # 计算平均角度
        # Calculate the average angle
        avg_angle = np.mean(angles)
        # 旋转图像
        # Rotate the image
        corrected_img = rotate_image(cropped_img, avg_angle)

        return corrected_img
    else:
        print("未检测到框选区域")
        print("No selected area detected.")
        return cropped_img

def rotate_image(image, angle):
    """
    旋转图像到指定角度
    Rotate the image to the specified angle.

    Args:
        image: 输入的图像
        image: The input image.
        angle: 旋转的角度
        angle: The rotation angle.

    Returns:
        numpy.ndarray: 旋转后的图像
        numpy.ndarray: The rotated image.
    """
    (h, w) = image.shape[:2]
    # 计算图像的中心点
    # Calculate the center point of the image
    center = (w // 2, h // 2)
    # 获取旋转矩阵
    # Get the rotation matrix
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    # 应用仿射变换进行旋转
    # Apply affine transformation for rotation
    rotated_img = cv2.warpAffine(image, M, (w, h))
    return rotated_img

def rotated_img_with_fft(gray):
    """
    使用傅里叶变换旋转图像。
    Rotate the image using Fourier transform.

    Args:
        gray: 输入的灰度图像
        gray: The input grayscale image.

    Returns:
        numpy.ndarray: 旋转后的图像
        numpy.ndarray: The rotated image.
    """
    # 图像延扩
    # Image extension
    gray_image = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    h, w = gray_image.shape[:2]
    new_h = cv2.getOptimalDFTSize(h)
    new_w = cv2.getOptimalDFTSize(w)
    right = new_w - w
    bottom = new_h - h
    nimg = cv2.copyMakeBorder(gray_image, 0, bottom, 0, right, borderType=cv2.BORDER_CONSTANT, value=0)

    # 执行傅里叶变换，并获得频域图像
    # Perform Fourier transform and obtain the frequency domain image
    f = np.fft.fft2(nimg)
    fshift = np.fft.fftshift(f)

    fft_img = np.log(np.abs(fshift))
    fft_img = (fft_img - np.amin(fft_img)) / (np.amax(fft_img) - np.amin(fft_img))

    fft_img *= 255
    ret, thresh = cv2.threshold(fft_img, 150, 255, cv2.THRESH_BINARY)

    # 霍夫直线变换
    # Hough line transform
    thresh = thresh.astype(np.uint8)
    lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, 30, minLineLength=40, maxLineGap=100)
    try:
        lines1 = lines[:, 0, :]
    except Exception as e:
        lines1 = []
    piThresh = np.pi / 180
    pi2 = np.pi / 2
    angle = 0
    for line in lines1:
        x1, y1, x2, y2 = line
        if x2 - x1 == 0:
            continue
        else:
            theta = (y2 - y1) / (x2 - x1)
        if abs(theta) < piThresh or abs(theta - pi2) < piThresh:
            continue
        else:
            angle = abs(theta)
            break

    angle = math.atan(angle)
    angle = angle * (180 / np.pi)
    print(angle)
    center = (w // 2, h // 2)
    height_1 = int(w * fabs(sin(radians(angle))) + h * fabs(cos(radians(angle))))
    width_1 = int(h * fabs(sin(radians(angle))) + w * fabs(cos(radians(angle))))
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += (width_1 - w) / 2
    M[1, 2] += (height_1 - h) / 2
    rotated = cv2.warpAffine(gray_image, M, (width_1, height_1), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    cv2.imshow('rotated', rotated)
    cv2.waitKey(0)
    return rotated

def rotated_img_with_radiation(gray, is_show=False):
    """
    使用辐射校正旋转图像。
    Rotate the image using radiation correction.

    Args:
        gray: 输入的灰度图像
        gray: The input grayscale image.
        is_show: 是否显示结果
        is_show: Whether to show the result.

    Returns:
        numpy.ndarray: 旋转后的图像
        numpy.ndarray: The rotated image.
    """
    gray_image = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    # 使用自适应阈值进行二值化
    # Use adaptive thresholding for binarization
    thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    if is_show:
        cv2.imshow('thresh', thresh)

    coords = np.column_stack(np.where(thresh > 0))
    # 计算最小外接矩形的角度
    # Calculate the angle of the minimum bounding rectangle
    angle = cv2.minAreaRect(coords)[-1]
    print(angle)

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # 旋转图像
    # Rotate the image
    rotated = rotate_image(gray_image, angle)

    if is_show:
        cv2.putText(rotated, 'Angle: {:.2f} degrees'.format(angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)
        cv2.imshow('Rotated', rotated)
        cv2.waitKey()

    return rotated

def get_angle_from_lines(lines):
    """
    计算霍夫变换得到的线的平均角度
    Calculate the average angle of the lines obtained by Hough transform.

    Args:
        lines: 霍夫变换得到的线
        lines: The lines obtained by Hough transform.

    Returns:
        float: 平均角度
        float: The average angle.
    """
    if lines is None or len(lines) == 0:
        return 0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # 计算直线的角度
        # Calculate the angle of the line
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        angles.append(angle)

    return np.mean(angles)

# 处理主调函数
# Main processing function
def process_image(cropped_img):
    """
    处理裁剪后的图像。
    Process the cropped image.

    Args:
        cropped_img: 裁剪后的图像
        cropped_img: The cropped image.

    Returns:
        numpy.ndarray: 处理后的图像
        numpy.ndarray: The processed image.
    """
    # 校正图像方向
    # Correct the image orientation
    corrected_img = correct_orientation(cropped_img)
    # 使用傅里叶变换旋转图像
    # Rotate the image using Fourier transform
    radiation_corrected_img = rotated_img_with_fft(corrected_img)

    return radiation_corrected_img

def process_image_file(img_path, save_Path, filename):
    """
    处理图像文件。
    Process the image file.

    Args:
        img_path: 图像文件路径
        img_path: The image file path.
        save_Path: 保存路径
        save_Path: The save path.
        filename: 文件名
        filename: The file name.

    Returns:
        numpy.ndarray: 处理后的图像
        numpy.ndarray: The processed image.
    """
    img_np = np.array(img_path)

    # 调整图像大小（根据模型设置调整）
    # Resize the image (adjust according to the model settings)
    img_resized = cv2.resize(img_np, (640, 640))  # 注意cv2.resize需要的是高宽顺序

    # 使用模型进行预测
    # Use the model for prediction
    results = model.predict(img_resized)
    img = np.array(img_resized)

    # 可视化结果
    # Visualize the results
    detections = results.xyxy[0].tolist()  # Convert results to a list of detections

    # Draw bounding boxes and labels on the image
    for *xyxy, conf, cls in detections:
        label = f'{results.names[int(cls)]} {conf:.2f}'  # 使用 results.names 而不是 model.names
        color = (255, 0, 0)  # Red (adjust as needed)
        plot_one_box(xyxy, img, label=label, color=color, line_thickness=3)
    if save_Path is not None:
        # 裁剪并保存图像
        # Crop and save the image
        imgs = crop_and_save(img, results.xyxy[0].tolist(), save_Path, filename)
    else:
        imgs = crop_and_save(img, results.xyxy[0].tolist())
    return imgs