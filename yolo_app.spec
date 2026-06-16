# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec 文件 — Ultralytics YOLO 智能检测平台

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# --- 收集 ultralytics 所有子模块和数据文件 ---
ultralytics_hidden_imports = collect_submodules('ultralytics')
ultralytics_datas = collect_data_files('ultralytics')

# --- 收集 torch 相关子模块 ---
torch_hidden_imports = collect_submodules('torch')
torch_datas = collect_data_files('torch', include_py_files=False)

a = Analysis(
    ['app_main.py'],
    pathex=[],
    binaries=[],
    datas=ultralytics_datas + torch_datas,
    hiddenimports=[
        # PySide6
        'PySide6',
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        'PySide6.QtNetwork',
        # OpenCV
        'cv2',
        # NumPy
        'numpy',
        # YAML
        'yaml',
        'yaml.loader',
        'yaml.cyaml',
        # SciPy (ultralytics 内部依赖)
        'scipy',
        'scipy.special',
        'scipy.ndimage',
        # PIL
        'PIL',
        'PIL.Image',
        # requests (模型下载)
        'requests',
        # tqdm (下载进度)
        'tqdm',
        # pandas
        'pandas',
        # matplotlib (可视化)
        'matplotlib',
        'matplotlib.pyplot',
        # Ultralytics 子模块
        'ultralytics',
        'ultralytics.nn',
        'ultralytics.nn.modules',
        'ultralytics.nn.modules.block',
        'ultralytics.nn.modules.conv',
        'ultralytics.nn.modules.head',
        'ultralytics.nn.modules.transformer',
        'ultralytics.cfg',
        'ultralytics.data',
        'ultralytics.engine',
        'ultralytics.models',
        'ultralytics.utils',
        'ultralytics.utils.autobatch',
        'ultralytics.utils.checks',
        'ultralytics.utils.downloads',
        'ultralytics.utils.metrics',
        'ultralytics.utils.ops',
        'ultralytics.utils.plotting',
        'ultralytics.utils.torch_utils',
        # Torch 核心
        'torch',
        'torch.nn',
        'torch.nn.functional',
        'torch.optim',
        'torch.cuda',
        'torch.backends',
        'torch.backends.cudnn',
        'torchvision',
        'torchvision.models',
        'torchvision.transforms',
    ] + ultralytics_hidden_imports + torch_hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'unittest',
        'pytest',
        'IPython',
        'jupyter',
        'notebook',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='YOLO_Smart_Detection',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,   # 正式发布时隐藏控制台窗口
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
