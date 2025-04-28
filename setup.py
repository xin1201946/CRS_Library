from setuptools import setup,find_packages
# python setup.py bdist_wheel
setup(
    name='CCRS_Library',
    version='2.0.6.9',
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
    description='CCRS_Library 是一个CCRS项目的必要组件，提供数据库快捷操作，剪裁和识别铸字图像功能。'
)
