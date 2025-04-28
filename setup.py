from setuptools import setup,find_packages
# python setup.py bdist_wheel
setup(
    name='CCRS_Library',
    version='2.0.6.10',
    packages=['CCRS_Library', 'CCRS_Library.TUI', 'CCRS_Library.sql', 'CCRS_Library.get_num', 'CCRS_Library.sys_info',
              'CCRS_Library.clear_pic'],
    url='',
    license='',
    install_requires=[
        'rich~=13.9.4',
        'opencv-python~=4.10.0.84',
        'ultralytics~=8.3.63',
        'setuptools~=70.0.0',
        'psutil~=6.1.1',
        'GPUtil~=1.4.0',
        'requests~=2.32.3',
        'torch~=2.6.0',
        'torchvision~=0.21.0',
        'torchaudio~=2.6.0',
        'wifi~=0.3.8',
        'matplotlib~=3.10.0',
        'yolov5~=7.0.14',
    ],
    package_data={
        'CCRS_Library.clear_pic': ['*.pt'],  # 包含所有.pt文件
        'CCRS_Library.get_num': ['*.pt'],  # 包含所有.pt文件
    },
    include_package_data=True,
    author='林间追风',
    author_email='1324435230@qq.com',
    description='CCRS_Library 是一个CCRS项目的必要组件，提供数据库快捷操作，剪裁和识别铸字图像功能。',
    long_description="CCRS_Library 是一个专注于轮毂铸造字符识别的开源 API 库，利用深度学习模型（YOLOv5 和 YOLOv11）提供高效的图像处理与字符识别功能。该库特别适用于制造业的智能质量检测，支持自动化生产线的集成。核心功能包括图像裁剪、字符识别（高准确性、灵活性和速度平衡版本）及数据库操作，具有模块化设计，方便快速部署。支持多种模型和图像处理功能，帮助提升铸造领域的自动化和精度。基于 GNU 宽松通用公共许可证 v3.0 (LGPLv3) 发布，允许自由使用、修改和分发。更多信息请访问 [GitHub - CCRS_Library](https://github.com/xin1201946/CRS_Library)。",
    long_description_content_type="text/plain"
)
