# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Sub-Team agent pipeline."""

import os
import sys

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('sub_team', 'sub_team'),
    ],
    hiddenimports=[
        'openai',
        'dotenv',
        'requests',
        'sub_team',
        'sub_team.specification_agent',
        'sub_team.microarchitecture_agent',
        'sub_team.implementation_agent',
        'sub_team.verification_agent',
        'sub_team.cross_disciplinary_agent',
        'sub_team.business_agent',
        'sub_team.llm_client',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name='SubTeam',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
