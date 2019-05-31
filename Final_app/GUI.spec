# -*- mode: python -*-
import sys
sys.setrecursionlimit(5000)

block_cipher = None


a = Analysis(['GUI.py'],
             pathex=['C:\\Users\\Aturt\\Documents\\music-sheet-generator\\Final_app'],
             binaries=[],
             datas=[],
             hiddenimports=['sklearn.neighbors.typedefs', 'sklearn.neighbors.quad_tree', 'sklearn.tree', 'sklearn.tree._utils'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='GUI',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='GUI')
