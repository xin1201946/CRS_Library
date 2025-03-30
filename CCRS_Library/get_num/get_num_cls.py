import os
import pathlib
import cv2
import numpy as np
import importlib.resources

# 临时存储PosixPath类
# Temporarily store the PosixPath class
temp = pathlib.PosixPath
# 在Windows系统上，将PosixPath替换为WindowsPath
# On Windows systems, replace PosixPath with WindowsPath
pathlib.PosixPath = pathlib.WindowsPath
# 初始化模型为None
# Initialize the model to None
model = None

def load_model():
    """
    加载YOLO模型。
    Load the YOLO model.

    Returns:
        YOLO: 加载好的YOLO模型。
        YOLO: The loaded YOLO model.
    """
    global model
    from ultralytics import YOLO
    # 使用importlib.resources获取模型文件的路径
    # Use importlib.resources to get the path of the model file
    with importlib.resources.path("CCRS_Library.get_num", "num_class_v2.pt") as model_path:
        # 加载模型
        # Load the model
        model = YOLO(str(model_path))
    return model

def compute_skew(image):
    """
    计算图像的倾斜角度。
    Calculate the skew angle of the image.

    Args:
        image (numpy.ndarray): 输入的图像。
        image (numpy.ndarray): The input image.

    Returns:
        float: 计算得到的倾斜角度。
        float: The calculated skew angle.
    """
    # 确保图像是灰度的
    # Ensure the image is grayscale
    if len(image.shape) == 3:
        # 如果是彩色图像，转换为灰度图像
        # If it is a color image, convert it to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 边缘检测
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # 霍夫线变换
    # Hough line transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    angles = []

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            # 只考虑接近水平或垂直的线
            # Only consider lines close to horizontal or vertical
            if theta < np.pi / 4 or theta > 3 * np.pi / 4:
                angles.append(theta)

    # 计算中位数角度
    # Calculate the median angle
    if len(angles) > 0:
        median_angle = np.median(angles)
        # 将角度转换为度数
        # Convert the angle to degrees
        skew_angle = np.rad2deg(median_angle - np.pi / 2)
        return skew_angle
    else:
        return 0

def deskew(image):
    """
    校正图像的倾斜。
    Correct the skew of the image.

    Args:
        image (numpy.ndarray): 输入的图像。
        image (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: 校正后的图像。
        numpy.ndarray: The corrected image.
    """
    # 反转图像以进行矩计算
    # Invert the image for moment calculation
    img = 255 - image

    # 计算矩
    # Calculate moments
    moments = cv2.moments(img)
    if abs(moments['mu02']) < 1e-2:
        return image

    # 计算倾斜度
    # Calculate skew
    skew = moments['mu11'] / moments['mu02']
    M = np.float32([[1, skew, -0.5 * img.shape[0] * skew],
                    [0, 1, 0]])

    # 应用仿射变换
    # Apply affine transform
    height, width = img.shape
    img = cv2.warpAffine(img, M, (width, height),
                         flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)

    # 反转回来
    # Invert back
    return 255 - img

def load_image(path, show_pic=False, show_lowPic=False):
    """
    加载并预处理图像。
    Load and preprocess the image.

    Args:
        path (str): 图像文件的路径。
        path (str): The path of the image file.
        show_pic (bool, optional): 是否显示最终处理结果。默认为False。
        show_pic (bool, optional): Whether to show the final processed result. Defaults to False.
        show_lowPic (bool, optional): 是否显示中间处理结果。默认为False。
        show_lowPic (bool, optional): Whether to show the intermediate processed result. Defaults to False.

    Returns:
        numpy.ndarray: 预处理后的图像。
        numpy.ndarray: The preprocessed image.
    """
    # 读取图像
    # Read the image
    image = cv2.imread(path)
    if image is None:
        raise ValueError("Error: Could not load image. Please check the file path.")
    # 调整图像大小
    # Resize the image
    resized_image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    # 转换为灰度图像
    # Convert to grayscale
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # 应用高斯模糊以减少噪声
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # 应用自适应阈值处理
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,  # 改为BINARY以使背景为白色
        # Changed to BINARY to make background white
        11,
        2
    )

    # 创建用于形态学操作的内核
    # Create kernels for morphological operations
    kernel_small = np.ones((2, 2), np.uint8)
    kernel_medium = np.ones((3, 3), np.uint8)

    # 清理噪声
    # Clean up noise
    denoised = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)

    # 填充数字中的小孔
    # Fill small holes in the digits
    filled = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel_medium)

    # 去除孤立像素
    # Remove isolated pixels
    cleaned = cv2.medianBlur(filled, 3)

    # 如果需要，调整大小（使用更好的插值方法）
    # Resize if needed (using better interpolation)
    if max(cleaned.shape) < 28:  # 良好识别的最小尺寸
        # Minimum size for good recognition
        scale = 28 / min(cleaned.shape)
        cleaned = cv2.resize(cleaned, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_LANCZOS4)

    # 应用轮廓过滤以去除小的伪影
    # Apply contour filtering to remove small artifacts
    contours, _ = cv2.findContours(255 - cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.ones_like(cleaned) * 255
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:  # 根据图像大小调整阈值
            # Adjust threshold based on your image size
            cv2.drawContours(mask, [contour], -1, 0, -1)

    final_image = cv2.bitwise_or(cleaned, mask)

    # 如果需要，校正倾斜
    # Deskew if needed
    final_image = deskew(final_image)

    if show_pic:
        if show_lowPic:
            # 显示原始灰度图像
            # Show the original grayscale image
            cv2.imshow('Original', gray_image)
            # 显示预处理后的图像
            # Show the preprocessed image
            cv2.imshow('After Preprocessing', binary)
        # 显示最终结果
        # Show the final result
        cv2.imshow('Final Result', final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return final_image

def save_files(path, save_name, img):
    """
    保存图像到指定路径。
    Save the image to the specified path.

    Args:
        path (str): 保存图像的目录路径。
        path (str): The directory path to save the image.
        save_name (str): 保存图像的文件名。
        save_name (str): The file name to save the image.
        img (numpy.ndarray): 要保存的图像。
        img (numpy.ndarray): The image to be saved.
    """
    if isinstance(path, np.ndarray):
        print("Error: path is an ndarray, converting to string")
        path = str(path)
    if not os.path.exists(path):
        # 如果目录不存在，则创建目录
        # Create the directory if it does not exist
        os.makedirs(path)
    full_path = os.path.join(path, save_name)
    print(f"Saving image to: {full_path}")
    # 保存图像
    # Save the image
    cv2.imwrite(full_path, img)

def main(save_path="./Out_PIC/", save_name='result.jpg', save_file=False, show_result=False, load_imagePath='./cache/crop.jpg'):
    """
    主函数，用于加载图像、进行预测并保存结果。
    The main function for loading images, making predictions, and saving results.

    Args:
        save_path (str, optional): 保存图像的路径。默认为"./Out_PIC/"。
        save_path (str, optional): The path to save the image. Defaults to "./Out_PIC/".
        save_name (str, optional): 保存图像的文件名。默认为'result.jpg'。
        save_name (str, optional): The file name to save the image. Defaults to 'result.jpg'.
        save_file (bool, optional): 是否保存图像。默认为True。
        save_file (bool, optional): Whether to save the image. Defaults to True.
        show_result (bool, optional): 是否显示识别结果。默认为True。
        show_result (bool, optional): Whether to show the recognition result. Defaults to True.
        load_imagePath (str, optional): 加载图像的路径。默认为'./cache/crop.jpg'。
        load_imagePath (str, optional): The path to load the image. Defaults to './cache/crop.jpg'.

    Returns:
        str: 识别出的类别名称。
        str: The recognized class name.
    """
    global model
    # 如果load_imagePath为None，则使用默认路径
    # If load_imagePath is None, use the default path
    img_path = './cache/crop.jpg' if load_imagePath is None else load_imagePath

    # 加载并预处理图像
    # Load and preprocess the image
    img = load_image(img_path, show_pic=show_result)

    if model is None:
        # 如果模型未加载，则加载模型
        # If the model is not loaded, load the model
        load_model()

    # 使用模型进行推理
    # Use the model for inference
    resized_image = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    results = model.predict(source=resized_image, save=False)

    class_name = ""
    confidence = 0.0
    for result in results:
        probs = result.probs
        class_id = probs.top1
        class_name = model.names[class_id]
        confidence = probs.top1conf.item()
        print(f"Recognized Class: {class_name}")
        print(f"Confidence: {confidence:.2f}")

    # 在图像上绘制结果
    # Draw the results on the image
    result_img = cv2.resize(img, (448, 448))  # 放大图像以便更好地显示
    # Enlarge the image for better display
    cv2.putText(result_img, f"Class: {class_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(result_img, f"Conf: {confidence:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if save_file:
        # 如果需要，保存图像
        # If needed, save the image
        save_files(save_path, save_name, cv2.resize(img, (448, 448)))

    if show_result:
        # 如果需要，显示识别结果
        # If needed, show the recognition result
        cv2.imshow('Recognition Result', result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 删除临时图像文件
    # Delete the temporary image file
    os.remove(img_path)

    return class_name