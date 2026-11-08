
# 受混沌侵蝕心智的龍王 — 泰拉瑞亞災厄風格像素繪（正面＋側面）
# 渲染流程：部件遮罩 → 依距離場算斜面光影（左上光源）→ 依材質套色階與紋理 → 部件間內描邊 → 外描邊 → 混沌（星空＋白紫外框＋發光）
import numpy as np, math
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

def hx(c): return tuple(int(c[i:i+2],16) for i in (1,3,5))
RAMPS={
 'scale':['#0b0f0c','#161d15','#222c1d','#313d26','#445231','#5c6c3e','#7b8a50'],
 'scaleD':['#080a08','#10150f','#192016','#232d1d','#313d26','#3e4a30','#4f5c38'],
 'belly':['#1e170c','#3a2e18','#5a4826','#7e6834','#a68c46','#cdb25c','#e8d27e'],
 'mem':['#120a06','#22140c','#361f12','#4e2e1a','#6a4024','#8a5630','#a86e3e'],
 'memD':['#0a0604','#140c07','#1e120b','#2c1a10','#3c2416','#4c2e1c','#5e3a24'],
 'bone':['#0c0806','#1a120c','#2a1e14','#3e2c1e','#56402a','#705638','#8c6e48'],
 'orange':['#2a0804','#5a1406','#8e260a','#c23e10','#e8641c','#ff9636','#ffcc6a'],
 'claw':['#1a0804','#4a1406','#8a2a0c','#d0501a','#ff8a34','#ffc066','#fff0b0'],
}
OUT=hx('#06050a')
class Sprite:
  def __init__(s,W,H): s.W,s.H=W,H; s.parts=[]
  def _new(s,mat,shift=0,tex=None):
    im=Image.new('L',(s.W,s.H),0); s.parts.append({'mat':mat,'im':im,'shift':shift,'tex':tex or mat}); return ImageDraw.Draw(im)
  def poly(s,mat,pts,**k): s._new(mat,**k).polygon([tuple(map(float,p)) for p in pts],fill=255)
  def ell(s,mat,cx,cy,rx,ry,**k): s._new(mat,**k).ellipse([cx-rx,cy-ry,cx+rx,cy+ry],fill=255)
  def tube(s,mat,pts,w0,w1,**k):
    d=s._new(mat,**k);n=len(pts)
    # 沿著折線取樣，畫一連串圓（漸細的管狀：脖子、尾巴、四肢、角）
    seg=[]
    for i in range(n-1):
      a,b=pts[i],pts[i+1];L=max(1,int(math.hypot(b[0]-a[0],b[1]-a[1])))
      for j in range(L): seg.append((a[0]+(b[0]-a[0])*j/L,a[1]+(b[1]-a[1])*j/L))
    seg.append(pts[-1]);m=len(seg)
    for j,(x,y) in enumerate(seg):
      r=w0+(w1-w0)*j/max(1,m-1); d.ellipse([x-r,y-r,x+r,y+r],fill=255)
  def spike(s,mat,base,tip,w,**k):
    bx,by=base;tx,ty=tip;dx,dy=tx-bx,ty-by;L=math.hypot(dx,dy);nx,ny=-dy/L,dx/L
    s.poly(mat,[(bx+nx*w,by+ny*w),(bx+dx*.55+nx*w*.75,by+dy*.55+ny*w*.75),(tx,ty),(bx+dx*.6-nx*w*.45,by+dy*.6-ny*w*.45),(bx-nx*w,by-ny*w)],**k)
  def render(s,glow=True,seed=7):
    W,H=s.W,s.H;N=len(s.parts);pid=np.full((H,W),-1,int)
    for i,p in enumerate(s.parts): pid[np.array(p['im'])>127]=i
    rgb=np.zeros((H,W,3),np.uint8);alpha=np.zeros((H,W),np.uint8);L=np.array([-1.,-1.])/math.sqrt(2)
    yy,xx=np.mgrid[0:H,0:W];rng=np.random.default_rng(seed)
    chaos=np.zeros((H,W),bool)
    for i,p in enumerate(s.parts):
      vis=pid==i
      if not vis.any(): continue
      if p['mat']=='chaos': chaos|=vis; continue
      full=np.array(p['im'])>127
      d=ndimage.distance_transform_edt(full);R=6.0;h=np.minimum(d,R)/R
      h=ndimage.gaussian_filter(h,1.0);gy,gx=np.gradient(h)
      lit=-(gx*L[0]+gy*L[1])*R*1.6
      ramp=[hx(c) for c in RAMPS[p['mat']]];n=len(ramp);base=n//2+p['shift']
      t=p['tex'];tex=np.zeros((H,W))
      if t in('scale','scaleD'):
        row=yy//4;u=(xx+(row%2)*3)%6;v=yy%4
        # 魚鱗狀：每片鱗下緣是弧形暗線，上緣中央有一點高光
        tex=np.where((v==3)&(u>=1)&(u<=4),-1.2,0)+np.where((v==2)&((u==0)|(u==5)),-1.0,0)+np.where((v==0)&(u>=2)&(u<=3),1.0,0)+np.where((v==1)&(u>=1)&(u<=4),.35,0)
      elif t=='belly': tex=np.where(yy%5==4,-1.5,0)+np.where(yy%5==0,1,0)
      elif t in('mem','memD'): tex=np.where(((xx+yy*3)%17==0),-.6,0)+np.sin(xx/5.0+yy/9.0)*.35
      elif t in('orange','claw'): tex=np.where(((xx-yy)%7==0),.8,0)
      elif t=='bone': tex=np.where(yy%3==0,-.5,0)
      idx=np.clip(np.round(base+lit+tex+(d<1.5)*-1.2),0,n-1).astype(int)
      cols=np.array(ramp)[idx];rgb[vis]=cols[vis];alpha[vis]=255
    body=pid>=0
    # 部件間內描邊：被前方部件壓住的邊緣變最暗
    for dy_,dx_ in((1,0),(-1,0),(0,1),(0,-1)):
      nb=np.roll(np.roll(pid,dy_,0),dx_,1)
      edge=body&(nb>pid)&(nb>=0)&~chaos
      for i,p in enumerate(s.parts):
        m=edge&(pid==i)
        if m.any(): rgb[m]=hx(RAMPS.get(p['mat'],RAMPS['scale'])[0])
    # 外描邊
    sil=body.copy();outline=ndimage.binary_dilation(sil)&~sil
    rgb[outline]=OUT;alpha[outline]=255
    # 混沌：星空內部＋白／紫外框＋發光
    if chaos.any():
      dc=ndimage.distance_transform_edt(chaos)
      k=((xx*13+yy*7)%9)/9.0
      sky=np.stack([18+k*14,10+k*8,40+k*34],-1).astype(np.uint8)
      rgb[chaos]=sky[chaos]
      hsh=(xx*73856093^yy*19349663)&0xffffffff
      st=chaos&(dc>2)&(hsh%19==0);rgb[st]=(235,230,255)
      st2=chaos&(dc>2)&(hsh%47==3);rgb[st2]=(190,150,255)
      st3=chaos&(dc>3)&(hsh%131==7);rgb[st3]=(255,255,255)
      rgb[chaos&(dc<=1)]=(255,255,255);rgb[chaos&(dc>1)&(dc<=2)]=(200,160,255);alpha[chaos]=255
      ring=ndimage.binary_dilation(chaos)&~chaos;rgb[ring]=(150,96,255);alpha[ring]=255
    img=Image.fromarray(np.dstack([rgb,alpha]),'RGBA')
    gl=None
    if glow and chaos.any():
      g=Image.fromarray((ndimage.binary_dilation(chaos,iterations=1)*255).astype(np.uint8),'L').filter(ImageFilter.GaussianBlur(5))
      ga=np.array(g).astype(float)/255;ga=np.clip(ga*1.6,0,1)
      gl=Image.fromarray(np.dstack([np.full((H,W),170,np.uint8),np.full((H,W),110,np.uint8),np.full((H,W),255,np.uint8),(ga*255).astype(np.uint8)]),'RGBA')
      # 眼睛也發光
    return img,gl

# ---------------- 正面 ----------------
def front():
  W,H=240,200;cx=120;GY=196;S=Sprite(W,H)
  def sym(f):
    f(lambda X:X,1);f(lambda X:2*cx-X,-1)
  # 尾巴（身後）
  S.tube('scaleD',[(cx-10,GY-36),(cx-46,GY-24),(cx-80,GY-18),(cx-104,GY-24)],7,1.5,shift=-1)
  for t in [(cx-50,GY-30),(cx-70,GY-25),(cx-88,GY-24)]: S.spike('orange',t,(t[0]-2,t[1]-9),2.5,shift=-1)
  # 翅膀
  def wing(X,s):
    sh=(X(cx-22),78);wr=(X(cx-66),40);tip=(X(cx-116),26);f=[(X(cx-118),70),(X(cx-110),104),(X(cx-92),132),(X(cx-66),142)]
    S.poly('mem',[sh,wr,tip,f[0],(X(cx-108),86),f[1],(X(cx-96),118),f[2],(X(cx-78),134),f[3],(X(cx-40),120)])
    S.poly('memD',[sh,wr,(X(cx-56),82),(X(cx-34),112)])
    for fp in f: S.tube('bone',[wr,fp],2.2,1.0)
    S.tube('bone',[sh,wr,tip],4.5,1.6)
    S.spike('claw',wr,(X(cx-60),20),3)
  sym(wing)
  # 腿
  def leg(X,s):
    S.tube('scale',[(X(cx-16),128),(X(cx-32),150)],10,7);S.tube('scale',[(X(cx-32),150),(X(cx-24),176)],6,4.5)
    S.spike('orange',(X(cx-32),150),(X(cx-46),148),4)
    S.poly('scale',[(X(cx-34),176),(X(cx-14),176),(X(cx-10),GY-2),(X(cx-38),GY-2)])
    for dx in(-14,-6,2,10): S.spike('claw',(X(cx-24+dx),GY-4),(X(cx-24+dx*1.25),GY+1),2)
  sym(leg)
  # 軀幹
  S.poly('scale',[(cx-24,82),(cx+24,82),(cx+28,100),(cx+22,134),(cx+12,146),(cx-12,146),(cx-22,134),(cx-28,100)])
  S.poly('belly',[(cx-11,86),(cx+11,86),(cx+13,138),(cx-13,138)])
  def ribs(X,s):
    for k in range(4): y=92+k*10;S.poly('scale',[(X(cx-13),y),(X(cx-27),y+3),(X(cx-25),y+9),(X(cx-13),y+7)],shift=1 if k%2==0 else 0)
  sym(ribs)
  # 手臂
  def arm(X,s):
    S.ell('scale',X(cx-26),86,10,10,shift=1)
    S.tube('scale',[(X(cx-26),88),(X(cx-44),114)],8,6);S.tube('scale',[(X(cx-44),114),(X(cx-46),138)],6,5)
    S.poly('orange',[(X(cx-42),108),(X(cx-62),118),(X(cx-58),140),(X(cx-48),136)])
    S.ell('scale',X(cx-46),140,6,5)
    for a in(-.6,-.2,.2,.6): S.spike('claw',(X(cx-46)+math.sin(a)*4*-s,144),(X(cx-46)+math.sin(a)*12*-s,156),2)
  sym(arm)
  # 肩上刃刺
  def sp(X,s):
    S.spike('orange',(X(cx-30),80),(X(cx-44),18),7);S.spike('orange',(X(cx-36),84),(X(cx-54),46),4,shift=-1)
  sym(sp)
  # 脖子＋頭
  S.tube('scale',[(cx,84),(cx,58)],10,8)
  S.poly('scale',[(cx-14,26),(cx+14,26),(cx+12,38),(cx+7,48),(cx+5,62),(cx-5,62),(cx-7,48),(cx-12,38)])
  S.poly('scale',[(cx-8,30),(cx+8,30),(cx+5,46),(cx+3,58),(cx-3,58),(cx-5,46)],shift=1)
  def headx(X,s):
    S.spike('scaleD',(X(cx-12),30),(X(cx-24),24),3)
    S.poly('scaleD',[(X(cx-7),48),(X(cx-13),56),(X(cx-7),64),(X(cx-4),60)])
    S.spike('claw',(X(cx-6),60),(X(cx-5),66),1.5)
    # 豎琴形角冠
    S.tube('orange',[(X(cx-9),28),(X(cx-17),18),(X(cx-21),8),(X(cx-19),0)],3,1.2)
    S.tube('orange',[(X(cx-16),18),(X(cx-26),14)],2,1,shift=-1);S.tube('orange',[(X(cx-19),10),(X(cx-27),4)],1.8,1,shift=1)
  sym(headx)
  # 混沌（右半身：畫面右側＝龍王的左側？此處依設定放在畫面右側）
  for a,l,w in[(-1.57,26,4.5),(-1.95,16,3),(-1.2,16,3)]: S.spike('chaos',(cx,24),(cx+math.cos(a)*l,24+math.sin(a)*l),w)
  S.poly('chaos',[(cx,68),(cx+9,77),(cx,90),(cx-9,77)])
  for a,l,w in[(-1.25,28,5.5),(-.75,22,4.5),(-1.75,18,4),(-.25,15,4)]: S.spike('chaos',(cx+30,86),(cx+30+math.cos(a)*l,86+math.sin(a)*l),w)
  S.poly('chaos',[(cx+40,110),(cx+54,114),(cx+56,134),(cx+44,136)])
  for px,py,a,l,w in[(cx+52,116,-.3,15,4),(cx+54,126,.2,15,4),(cx+50,136,.8,11,3)]: S.spike('chaos',(px,py),(px+math.cos(a)*l,py+math.sin(a)*l),w)
  S.poly('chaos',[(cx+66,96),(cx+98,78),(cx+110,104),(cx+94,124),(cx+70,120)])
  for px,py,a,l,w in[(cx+100,84,-1.1,18,4.5),(cx+108,104,-.2,16,4),(cx+92,124,.9,15,4),(cx+76,120,1.4,12,3)]: S.spike('chaos',(px,py),(px+math.cos(a)*l,py+math.sin(a)*l),w)
  S.spike('chaos',(cx+28,170),(cx+40,166),3)
  # 眼睛（紫白光）
  for X in(cx-8,cx+8): S.ell('chaos',X,34,3,1.6)
  return S

# ---------------- 側面（面向左）----------------
def side():
  W,H=300,210;GY=206;S=Sprite(W,H)
  # 遠側翅膀（較暗）
  sh=(150,80);wr=(116,28);tip=(54,14);f=[(52,58),(70,88),(100,106),(130,110)]
  S.poly('memD',[sh,wr,tip,f[0],(62,72),f[1],(84,98),f[2],(116,110),f[3]])
  for fp in f: S.tube('bone',[wr,fp],2,1,shift=-2)
  S.tube('bone',[sh,wr,tip],4,1.5,shift=-2)
  # 尾巴
  S.tube('scale',[(176,140),(212,152),(246,166),(276,182),(292,196)],12,2)
  for i,t in enumerate([(200,140),(222,150),(244,160),(264,172),(280,184)]): S.spike('orange',t,(t[0]+5,t[1]-12+i),3.5-i*.4)
  # 遠側腳
  S.tube('scaleD',[(170,136),(188,160),(176,186)],9,5,shift=-1)
  S.poly('scaleD',[(170,186),(184,186),(178,GY-2),(156,GY-2)],shift=-1)
  for dx in(0,8,15): S.spike('claw',(158+dx,GY-4),(152+dx,GY+1),2,shift=-1)
  # 軀幹
  S.poly('scale',[(122,86),(148,82),(176,104),(188,132),(176,148),(150,146),(128,126),(116,104)])
  S.poly('belly',[(122,98),(130,92),(150,132),(160,146),(150,148),(128,126)])
  for k in range(6): y=100+k*8;x0=124+k*5;S.poly('scale',[(x0+14,y-6),(x0+28,y-4),(x0+30,y+2),(x0+16,y+1)],shift=1 if k%2 else 0)
  # 背脊刺
  for t in[(150,84),(164,92),(176,104),(184,118)]: S.spike('orange',t,(t[0]+6,t[1]-12),3)
  # 近側腳（逆關節）
  S.ell('scale',166,136,16,14,shift=1)
  S.tube('scale',[(168,140),(188,166)],10,7);S.tube('scale',[(188,166),(170,190)],7,5)
  S.spike('orange',(188,166),(202,162),4)
  S.poly('scale',[(160,188),(178,188),(172,GY-2),(140,GY-2)])
  for dx in(0,9,17): S.spike('claw',(144+dx,GY-4),(136+dx,GY+2),2.2)
  # 脖子（S 形）＋頭（朝左）
  S.tube('scale',[(130,92),(118,72),(104,58),(88,50)],11,8)
  for k,(px,py) in enumerate([(124,80),(112,66),(98,56)]): S.spike('orange',(px+4,py-6),(px+10,py-16),2.5)
  S.poly('scale',[(98,38),(84,34),(62,40),(40,48),(36,54),(56,58),(78,58),(96,54)])
  S.poly('scale',[(92,40),(80,38),(62,44),(46,50),(60,52),(80,52)],shift=1)
  S.poly('scaleD',[(88,56),(70,60),(50,62),(42,58),(60,64),(84,66)])
  S.spike('claw',(44,55),(38,62),2);S.spike('claw',(54,60),(50,66),1.6);S.spike('claw',(66,61),(63,67),1.6)
  S.spike('scaleD',(80,36),(92,26),3)
  # 角冠（往後上方掃）
  S.tube('orange',[(86,36),(98,24),(110,14),(124,10)],3.5,1.2)
  S.tube('orange',[(96,26),(104,12)],2,1,shift=1);S.tube('orange',[(106,16),(118,2)],1.8,1)
  S.tube('orange',[(92,40),(112,34),(126,30)],2.2,1,shift=-1)
  # 近側手臂
  S.ell('scale',134,100,11,10,shift=1)
  S.tube('scale',[(132,102),(114,122)],8,6);S.tube('scale',[(114,122),(96,128)],6,5)
  S.poly('orange',[(118,116),(126,132),(104,138),(98,132)])
  S.ell('scale',94,129,6,5)
  for a in(-.4,0,.4): S.spike('claw',(90,130+a*4),(80,136+a*10),2)
  # 肩刃
  S.spike('orange',(140,88),(164,26),7);S.spike('orange',(132,92),(146,50),4,shift=-1)
  # 近側翅膀（舉起，往後展）
  sh2=(146,84);wr2=(176,26);tip2=(262,10);g=[(270,52),(250,84),(222,102),(190,108)]
  S.poly('mem',[sh2,wr2,tip2,g[0],(262,68),g[1],(238,94),g[2],(206,106),g[3],(160,104)])
  S.poly('memD',[sh2,wr2,(186,70),(170,100)])
  for fp in g: S.tube('bone',[wr2,fp],2.2,1.0)
  S.tube('bone',[sh2,wr2,tip2],4.5,1.6)
  S.spike('claw',wr2,(170,8),3)
  # 混沌（側面看到的是被侵蝕的那一側）
  S.poly('chaos',[(206,48),(238,40),(252,66),(232,90),(208,84)])
  for px,py,a,l,w in[(240,44,-1.0,18,4.5),(252,66,-.2,16,4),(232,90,.8,15,4),(210,84,1.6,12,3)]: S.spike('chaos',(px,py),(px+math.cos(a)*l,py+math.sin(a)*l),w)
  for a,l,w in[(-1.6,26,5.5),(-1.1,20,4.5),(-2.1,16,4),(-.6,13,3.5)]: S.spike('chaos',(140,90),(140+math.cos(a)*l,90+math.sin(a)*l),w)
  S.poly('chaos',[(110,116),(122,122),(112,134),(102,130)])
  for px,py,a,l,w in[(112,118,-1.9,14,4),(104,128,2.6,12,3.5),(118,130,1.9,10,3)]: S.spike('chaos',(px,py),(px+math.cos(a)*l,py+math.sin(a)*l),w)
  for a,l,w in[(-1.7,22,4.5),(-2.2,14,3),(-1.2,14,3)]: S.spike('chaos',(86,34),(86+math.cos(a)*l,34+math.sin(a)*l),w)
  S.spike('chaos',(176,178),(190,174),3)
  S.ell('chaos',66,44,3,1.6)
  return S

import json,base64,io
out={}
for name,fn in[('front',front),('side',side)]:
  img,gl=fn().render()
  img.save(f'dragon_{name}.png')
  if gl: gl.save(f'dragon_{name}_glow.png')
  # 預覽：深色背景＋發光＋放大 4 倍
  W,H=img.size;bg=Image.new('RGBA',(W,H),(22,18,30,255))
  if gl: bg=Image.alpha_composite(bg,gl)
  bg=Image.alpha_composite(bg,img);bg=bg.resize((W*4,H*4),Image.NEAREST);bg.save(f'dragon_{name}_preview.png')
  for k,im in[('img',img),('glow',gl)]:
    b=io.BytesIO();im.save(b,'PNG');out[name+'_'+k]='data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()
json.dump(out,open('dragon_data.json','w'))
print('ok')
