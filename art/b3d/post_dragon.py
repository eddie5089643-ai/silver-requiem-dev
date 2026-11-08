
# 後製：Blender 三張圖（材質 ID／明暗／深度）→ 災厄風格像素圖＋發光圖層
import numpy as np, json, io, base64, sys
from PIL import Image, ImageFilter
from scipy import ndimage
B='/home/claude/sr_music/b3d'
def hx(c): return np.array([int(c[i:i+2],16) for i in (1,3,5)])
RAMPS={
 'scale':['#0b0f0c','#161d15','#222c1d','#313d26','#445231','#5c6c3e','#7b8a50'],
 'belly':['#1e170c','#3a2e18','#5a4826','#7e6834','#a68c46','#cdb25c','#e8d27e'],
 'mem':['#120a06','#22140c','#361f12','#4e2e1a','#6a4024','#8a5630','#a86e3e'],
 'bone':['#0c0806','#1a120c','#2a1e14','#3e2c1e','#56402a','#705638','#8c6e48'],
 'orange':['#2a0804','#5a1406','#8e260a','#c23e10','#e8641c','#ff9636','#ffcc6a'],
 'claw':['#1a0804','#4a1406','#8a2a0c','#d0501a','#ff8a34','#ffc066','#fff0b0'],
}
MATS=['scale','belly','mem','bone','orange','claw','chaos','eye']
def srgb(c):
  c=c/255.0;return np.where(c<=.0031308,c*12.92,1.055*c**(1/2.4)-.055)*255
ID_COL={m:srgb(np.array([((i+1)*29%256),((i*83+40)%256),((i*151+90)%256)],float)) for i,m in enumerate(MATS)}
def proc(tag):
  idi=np.array(Image.open(f'{B}/{tag}_id.png').convert('RGBA')).astype(int);lum=np.array(Image.open(f'{B}/{tag}_lum.png').convert('RGBA')).astype(float)
  dep=np.array(Image.open(f'{B}/{tag}_depth.png').convert('RGBA')).astype(float)
  H,W=idi.shape[:2];a=idi[...,3]>128
  # 材質編號：找最接近的 ID 顏色
  cols=np.stack([ID_COL[m] for m in MATS]);d=((idi[...,None,:3]-cols[None,None])**2).sum(-1);mid=np.argmin(d,-1);mid[~a]=-1
  # 清掉單一像素雜點（像素畫不要孤立點）
  for _ in range(1):
    md=ndimage.generic_filter(mid,lambda v:np.bincount((v+1).astype(int)).argmax()-1,size=3,mode='nearest');iso=(ndimage.generic_filter((mid[...,]),lambda v:(v==v[4]).sum(),size=3)<=2)&(mid>=0);mid[iso]=md[iso]
  L=lum[...,0]/255.0;z=dep[...,0]
  lo,hi=np.percentile(L[a],3),np.percentile(L[a],99.5);L=np.clip((L-lo)/max(.05,hi-lo),0,1)  # 每張圖自動拉滿明暗範圍
  rgb=np.zeros((H,W,3),np.uint8);out_a=np.zeros((H,W),np.uint8)
  yy,xx=np.mgrid[0:H,0:W]
  for i,m in enumerate(MATS):
    sel=mid==i
    if not sel.any() or m in('chaos','eye'):continue
    R=[hx(c) for c in RAMPS[m]];n=len(R)
    # 明暗量化（稍微拉對比）＋輕微有序抖動，讓過渡像手點的
    bayer=np.array([[0,2],[3,1]])/4.0-.375
    Lm=L[sel];lo2,hi2=np.percentile(Lm,4),np.percentile(Lm,99);Ls=np.clip((L-lo2)/max(.08,hi2-lo2),0,1)*.86+.1  # 每種材質各自拉明暗範圍
    v=Ls**1.05*(n-1)+bayer[yy%2,xx%2]*.55
    idx=np.clip(np.round(v),0,n-1).astype(int)
    rgb[sel]=np.array(R)[idx][sel];out_a[sel]=255
  # 內描邊：被前面部件擋住的邊緣（深度較遠的那一側）
  inner=np.zeros((H,W),bool)
  for dy,dx in((1,0),(-1,0),(0,1),(0,-1)):
    nz=np.roll(np.roll(z,dy,0),dx,1);nm=np.roll(np.roll(mid,dy,0),dx,1)
    inner|=(mid>=0)&(nm>=0)&(z>nz+3.0)
  for i,m in enumerate(MATS):
    if m in RAMPS:
      s=inner&(mid==i);rgb[s]=hx(RAMPS[m][0])
  body=mid>=0;chaos=(mid==MATS.index('chaos'));eye=mid==MATS.index('eye')
  # 外描邊
  ol=ndimage.binary_dilation(body)&~body;rgb[ol]=(6,5,10);out_a[ol]=255
  # 混沌：星空＋白／紫外框
  if chaos.any():
    dc=ndimage.distance_transform_edt(chaos);k=((xx*13+yy*7)%9)/9.0
    sky=np.stack([18+k*14,10+k*8,40+k*34],-1).astype(np.uint8);rgb[chaos]=sky[chaos]
    hsh=(xx*73856093^yy*19349663)&0xffffffff
    rgb[chaos&(dc>2)&(hsh%17==0)]=(235,230,255);rgb[chaos&(dc>2)&(hsh%43==3)]=(190,150,255)
    rgb[chaos&(dc<=1)]=(255,255,255);rgb[chaos&(dc>1)&(dc<=2)]=(200,160,255);out_a[chaos]=255
    ring=ndimage.binary_dilation(chaos)&~chaos&~body|((ndimage.binary_dilation(chaos)&~chaos)&body);rgb[ring]=(150,96,255);out_a[ring]=255
  rgb[eye]=(235,215,255);out_a[eye]=255
  img=Image.fromarray(np.dstack([rgb,out_a]),'RGBA')
  gm=(chaos|eye).astype(np.uint8)*255;g=np.array(Image.fromarray(gm).filter(ImageFilter.GaussianBlur(5))).astype(float)/255;g=np.clip(g*1.7,0,1)
  gl=Image.fromarray(np.dstack([np.full((H,W),170,np.uint8),np.full((H,W),110,np.uint8),np.full((H,W),255,np.uint8),(g*255).astype(np.uint8)]),'RGBA')
  return img,gl
meta=json.load(open(f'{B}/meta.json'));out={};anch={}
tags=['front','q3','side']+[f'fly{k}' for k in range(6)]
res={t:proc(t) for t in tags}
# 裁切：飛行 6 幀用同一個框（錨點才不會跳動）
def bbox(im):
  a=np.array(im)[...,3];ys,xs=np.nonzero(a);return xs.min(),ys.min(),xs.max()+1,ys.max()+1
groups=[['front'],['q3'],['side'],[f'fly{k}' for k in range(6)]]
for gp in groups:
  bb=[bbox(res[t][0]) for t in gp];x0=max(0,min(b[0] for b in bb)-8);y0=max(0,min(b[1] for b in bb)-8);x1=max(b[2] for b in bb)+8;y1=max(b[3] for b in bb)+8
  for t in gp:
    im,gl=res[t];im=im.crop((x0,y0,x1,y1));gl=gl.crop((x0,y0,x1,y1));im.save(f'{B}/dk_{t}.png');gl.save(f'{B}/dk_{t}_glow.png')
    ax,ay=meta[t]['anchor'];anch[t]=[round(ax-x0),round(ay-y0)]
    W,H=im.size;bg=Image.new('RGBA',(W,H),(22,18,30,255));bg=Image.alpha_composite(Image.alpha_composite(bg,gl),im).resize((W*4,H*4),Image.NEAREST);bg.save(f'{B}/dk_{t}_preview.png')
    for k,x in(('img',im),('glow',gl)):
      bio=io.BytesIO();x.save(bio,'PNG');out[t+'_'+k]='data:image/png;base64,'+base64.b64encode(bio.getvalue()).decode()
out['anchor']=anch;json.dump(out,open(f'{B}/dk_data.json','w'));print('ok',anch)
