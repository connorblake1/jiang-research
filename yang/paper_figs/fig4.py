import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
from matplotlib import font_manager
font_dirs = ["/home/cjb/yang/paper_figs"]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
  print(f"Adding {font_file}")
  font_manager.fontManager.addfont(font_file)
  print("Added.")

df = pd.read_excel("PassivePlusActiveTraining.xlsx",dtype={'substrate':'category','wavelength':'category'})
substrates = df.substrate.unique()
target_wav = int(sys.argv[1]) #514
Rc_wav = int(sys.argv[2]) # 514
dfs = [df[(df.substrate == value)&(df.wavelength == target_wav)] for value in substrates]
R_targets = [.2,.5,.8]
R_colors = ['#00a3a4','#00bca1','#00d493','#69e882','#acfa70']
R_colors = ['red','blue','green']
csfont = {'fontname':'Arial'}
hfont = {'fontname':'Arial'}
curves = []
zcap = 10000
for i,dfi in enumerate(dfs):
  T = np.round(dfi.iloc[0]['temperature'])
  R_c = np.round(dfi.iloc[0][str(Rc_wav) + "Rc"],4)
#print(substrates[i],T,R_c)

  times = dfi['time'].to_numpy().flatten()
  R = dfi['R'].to_numpy().flatten()
  zcap_cutoff_indices = np.where(times <= zcap)[0]
  # times = times[zcap_cutoff_indices]
  # R = times[zcap_cutoff_indices]
  # print(times,R)

  R_indices = [np.argmin(np.abs(R-R_targ)) for R_targ in R_targets]
  # print(R_indices)
  # print(dfi.head())
#print()
  curve = {
    "name":substrates[i],
    "R_c":R_c,
    "T":T,
    "t":times,
    "R":R,
    "R_i":R_indices
  }
  curves.append(curve)



# plotting
ax = plt.figure().add_subplot(projection="3d")

for curve in curves:
  nt = len(curve["t"])
  T = np.full((nt,1),curve["T"]).flatten()
  R_c = np.full((nt,1),curve["R_c"]).flatten()
  t = curve["t"]
  zcapi = np.where(t <= zcap)[0][-1]
  print(zcap_cutoff_indices)
  ax.plot(T[:zcapi],R_c[:zcapi],t[:zcapi],color='black')
fsize = 14
for i,R_targ in enumerate(R_targets):
  x = []
  y = []
  z = []
  for sub in curves:
    tcut = sub['t'][sub["R_i"][i]]
    Tcut = sub["T"]
    Rccut = np.array(sub["R_c"])
    x.append(Tcut)
    y.append(Rccut)
    z.append(tcut)

  x = np.array(x).flatten()
  y = np.array(y).flatten()
  z = np.array(z).flatten()


  print(np.where(x==820)[0])
  yi = 10#np.argmin(y[np.where(x == 820)[0]])
  ax.text(x[yi],y[yi]-.02,z[yi],f"R = {R_targ}",color=R_colors[i],fontsize=fsize,fontname="Arial",horizontalalignment='right',verticalalignment='center')
  ax.scatter(x,y,z,color=R_colors[i],marker='o')
  ax.plot_trisurf(x,y,z,linewidth=0,edgecolor=R_colors[i],antialiased=True,alpha=.1,color=R_colors[i])

ax.tick_params(axis='x', labelsize=fsize)
ax.tick_params(axis='y', labelsize=fsize)
ax.tick_params(axis='z', labelsize=fsize)
ax.set_xlabel("Temperature (C)",fontsize=fsize,fontname="Arial")
ax.set_ylabel("$R_{c} @ " + str(Rc_wav) + "$",fontsize=fsize,fontname="Arial")
ax.set_zlabel("Time (s)",fontsize=fsize,fontname="Arial")
for label in ax.get_xticklabels():
  label.set_fontname("Arial")
for label in ax.get_yticklabels():
  label.set_fontname("Arial")
for label in ax.get_zticklabels():
  label.set_fontname("Arial")
ax.xaxis.labelpad = 10
ax.yaxis.labelpad = 10
ax.zaxis.labelpad = 10
ax.set_zlim([0,zcap])
ax.set_title("Level Sets of $R_{" + str(target_wav) + "}$",fontsize=fsize+6,fontname="Arial")
plt.show()
  
  
  





