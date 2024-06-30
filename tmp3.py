import matplotlib.font_manager as font_manager
import os

dirs_fonts = [os.path.join('public','luqiqi','fonts','winsfonts'),]
font_files = font_manager.findSystemFonts(fontpaths=dirs_fonts)
font_list = font_manager.createFontList(font_files)
font_manager.fontManager.ttflist.extend(font_list)

a = sorted([f.name for f in font_manager.fontManager.ttflist])
for i in a:
    print(i)

from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

fig, ax = plt.subplots(dpi=300)
fpath = Path(os.path.join('fonts','times.ttf'))
ax.set_title(f'This is a special font: {fpath.name}', font=fpath)
ax.set_xlabel('This is the default font')
plt.savefig('figures/tmp')