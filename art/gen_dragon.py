
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
  def __init__(s,W,H,ox=0,oy=0): s.W,s.H=W,H; s.parts=[]; s.ox,s.oy=ox,oy
  def _new(s,mat,shift=0,tex=None):
    im=Image.new('L',(s.W,s.H),0); s.parts.append({'mat':mat,'im':im,'shift':shift,'tex':tex or mat}); return ImageDraw.Draw(im)
  def poly(s,mat,pts,**k): s._new(mat,**k).polygon([(float(p[0])+s.ox,float(p[1])+s.oy) for p in pts],fill=255)
  def ell(s,mat,cx,cy,rx,ry,**k): cx+=s.ox;cy+=s.oy;s._new(mat,**k).ellipse([cx-rx,cy-ry,cx+rx,cy+ry],fill=255)
  def tube(s,mat,pts,w0,w1,**k):
    d=s._new(mat,**k);pts=[(p[0]+s.ox,p[1]+s.oy) for p in pts];n=len(pts)
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


def lerp2(a,b,t): return (a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t)
def wing(S,sh,th,R1,R2,mat,bone_shift=0,chaos=False,flip=1):
  # th：翼腕相對肩膀的角度；指骨由翼尖往身體方向張開
  wr=(sh[0]+math.cos(th)*R1*flip,sh[1]+math.sin(th)*R1)
  td=th-0.35*flip
  tip=(wr[0]+math.cos(td)*R2*flip,wr[1]+math.sin(td)*R2)
  fs=[]
  for k,(da,L) in enumerate([(0.55,.92),(1.05,.8),(1.55,.68),(2.0,.55)]):
    a=td+da*flip;fs.append((wr[0]+math.cos(a)*R2*L*flip,wr[1]+math.sin(a)*R2*L))
  mid=lerp2(sh,fs[-1],.5)
  pts=[sh,wr,tip]
  for k,f in enumerate(fs):
    prev=tip if k==0 else fs[k-1];m=lerp2(prev,f,.5);dx,dy=m[0]-wr[0],m[1]-wr[1];L=math.hypot(dx,dy) or 1
    pts.append((m[0]-dx/L*R2*.12,m[1]-dy/L*R2*.12));pts.append(f)
  pts.append(lerp2(sh,fs[-1],.55))
  S.poly(mat,pts)
  S.poly('memD' if mat=='mem' else mat,[sh,wr,lerp2(wr,fs[2],.45),lerp2(sh,fs[-1],.4)],shift=0)
  for f in fs: S.tube('bone',[wr,f],2.2,1.0,shift=bone_shift)
  S.tube('bone',[sh,wr,tip],4.5,1.6,shift=bone_shift)
  S.spike('claw',wr,(wr[0]+math.cos(th-1.2*flip)*12*flip,wr[1]+math.sin(th-1.2*flip)*12),3,shift=bone_shift)
  if chaos:
    S.poly('chaos',[lerp2(wr,tip,.45),lerp2(wr,fs[0],.75),lerp2(wr,fs[1],.8),lerp2(wr,fs[1],.4)])
    for f,(a,l) in zip([lerp2(wr,tip,.5),lerp2(wr,fs[0],.78),lerp2(wr,fs[1],.8)],[(-1.0,16),(-.2,14),(.8,13)]):
      S.spike('chaos',f,(f[0]+math.cos(td+a)*l*flip,f[1]+math.sin(td+a)*l),4)
  return wr,tip,fs

# ---------------- 45 度（面向左前方）----------------
def three_q(k=None,N=6):
  if k is None: W,H=270,220;S=Sprite(W,H);thn=-0.75
  else:
    ph=k/N*2*math.pi;f=(1-math.cos(ph))/2;thn=-1.45+f*1.75;bob=round(5*math.cos(ph))
    W,H=300,350;S=Sprite(W,H,15,60+bob)
  GY=210;thf=-2.35-(thn+0.75)*.95
  wing(S,(112,86),thf,46,62,'memD',bone_shift=-2,flip=1)
  # 尾巴：靜態時垂到地面再往後拖；懸停時整條往下垂、尾端微微擺動並捲起
  if k is None: tp=[(150,168),(172,186),(188,204),(214,208),(244,206),(262,198)]
  else:
    sw=math.sin(ph+1.2)*6;tp=[(150,168),(168,190),(178,214),(182+sw*.4,238),(178+sw*.8,258),(168+sw,272),(158+sw,276)]
  S.tube('scaleD',tp,10,2,shift=-1)
  for i in range(1,len(tp)-2):
    a,c=tp[i],tp[i+1];dx,dy=c[0]-a[0],c[1]-a[1];L=math.hypot(dx,dy) or 1;nx,ny=dy/L,-dx/L
    if nx<0:nx,ny=-nx,-ny
    m=((a[0]+c[0])/2,(a[1]+c[1])/2);S.spike('orange',m,(m[0]+nx*10,m[1]+ny*10),3-i*.3,shift=-1)
  S.tube('scaleD',[(110,140),(96,162)],9,6,shift=-1);S.tube('scaleD',[(96,162),(104,186)],6,4,shift=-1)
  S.poly('scaleD',[(92,186),(112,186),(108,GY-2),(84,GY-2)],shift=-1)
  for dx in(0,8,16): S.spike('claw',(86+dx,GY-4),(80+dx,GY+1),2,shift=-1)
  S.spike('orange',(104,90),(88,38),5,shift=-1)
  S.tube('scaleD',[(102,94),(86,118)],7,5,shift=-1);S.tube('scaleD',[(86,118),(84,138)],5,4,shift=-1)
  S.poly('orange',[(84,112),(72,124),(76,140),(84,134)],shift=-1)
  for a in(-.4,0,.4): S.spike('claw',(84,140),(78+a*10,150),1.8,shift=-1)
  S.poly('scale',[(98,86),(146,84),(162,104),(158,140),(140,154),(108,152),(96,134),(92,104)])
  S.poly('belly',[(98,92),(124,90),(128,148),(106,146),(96,128)])
  for k in range(5): y=96+k*10;S.poly('scale',[(130,y),(156,y+2),(156,y+8),(130,y+7)],shift=1 if k%2 else 0)
  S.ell('scale',148,140,16,14,shift=1)
  S.tube('scale',[(150,144),(168,164)],10,7);S.tube('scale',[(168,164),(156,188)],7,5)
  S.spike('orange',(168,164),(182,160),4)
  S.poly('scale',[(146,188),(164,188),(160,GY-2),(132,GY-2)])
  for dx in(0,9,17): S.spike('claw',(136+dx,GY-4),(128+dx,GY+2),2.2)
  S.tube('scale',[(120,90),(110,62),(102,48)],10,8)
  S.poly('scale',[(116,32),(122,42),(116,56),(100,64),(82,62),(70,54),(78,44),(96,34)])
  S.poly('scale',[(112,36),(116,44),(110,54),(98,58),(84,56),(78,50),(96,40)],shift=1)
  S.poly('scaleD',[(110,58),(96,66),(80,66),(72,58),(84,62),(100,62)])
  S.spike('claw',(74,56),(68,62),1.8);S.spike('claw',(84,63),(80,69),1.5)
  S.tube('orange',[(110,36),(118,24),(124,12),(132,6)],3.2,1.2);S.tube('orange',[(118,24),(130,22)],1.8,1,shift=1)
  S.tube('orange',[(96,36),(92,24),(86,14),(80,10)],2.4,1,shift=-1)
  S.ell('scale',152,96,11,10,shift=1)
  S.tube('scale',[(152,98),(166,122)],8,6);S.tube('scale',[(166,122),(156,142)],6,5)
  S.poly('orange',[(166,114),(182,124),(172,146),(160,138)])
  S.ell('scale',154,144,6,5)
  for a in(-.4,0,.4): S.spike('claw',(150,146),(142+a*10,158),2)
  S.spike('orange',(152,88),(172,26),7);S.spike('orange',(160,94),(182,52),4,shift=-1)
  wing(S,(150,88),thn,52,70,'mem',flip=1,chaos=True)
  S.poly('chaos',[(112,96),(120,104),(112,116),(104,104)])
  for a,l,w in[(-1.3,26,5.5),(-.8,20,4.5),(-1.8,16,4),(-.3,13,3.5)]: S.spike('chaos',(156,92),(156+math.cos(a)*l,92+math.sin(a)*l),w)
  S.poly('chaos',[(162,116),(176,122),(170,138),(158,134)])
  for px,py,a,l,w in[(172,120,-.3,14,4),(174,132,.3,13,4)]: S.spike('chaos',(px,py),(px+math.cos(a)*l,py+math.sin(a)*l),w)
  for a,l,w in[(-1.6,22,4.5),(-2.0,14,3),(-1.2,14,3)]: S.spike('chaos',(104,34),(104+math.cos(a)*l,34+math.sin(a)*l),w)
  S.spike('chaos',(164,180),(178,176),3)
  S.ell('chaos',104,46,3,1.6);S.ell('chaos',88,46,2,1.2)
  return S

# ---------------- 振翅飛行（側面，面向左，6 幀）----------------
def fly(k,N=6):
  W,H=320,260;S=Sprite(W,H);ph=k/N*2*math.pi;f=(1-math.cos(ph))/2  # 0=翅膀最高，1=最低
  th=-1.8+f*2.55;bob=round(5*math.cos(ph));oy=70+bob
  def P(x,y): return (x,y+oy)
  wing(S,P(160,78),th-0.12,48,70,'memD',bone_shift=-2,flip=1)
  sw=math.sin(ph)*4
  S.tube('scale',[P(196,104),P(226,110),P(256,116+sw*.5),P(284,114+sw),P(306,106+sw*1.5)],11,2)
  for i,t in enumerate([P(214,104),P(236,108),P(258,112+sw*.5),P(280,110+sw)]): S.spike('orange',t,(t[0]+8,t[1]-9),3.2-i*.4)
  S.tube('scaleD',[P(186,112),P(202,128),P(222,132)],7,4,shift=-1)
  for dx in(0,6): S.spike('claw',P(222+dx,132),P(230+dx,136),1.8,shift=-1)
  S.poly('scale',[P(118,86),P(146,74),P(184,86),P(204,104),P(196,120),P(160,118),P(128,108)])
  S.poly('belly',[P(122,96),P(136,104),P(164,116),P(196,120),P(186,110),P(150,104),P(128,92)])
  for t in[P(150,76),P(166,80),P(182,88),P(196,98)]: S.spike('orange',t,(t[0]+9,t[1]-9),3)
  S.ell('scale',*P(184,112),15,11,shift=1)
  S.tube('scale',[P(188,116),P(208,128)],8,6);S.tube('scale',[P(208,128),P(226,126)],5,4)
  for dx in(0,6,11): S.spike('claw',P(226,124+dx*.3),P(236,128+dx*.4),2)
  S.tube('scale',[P(124,90),P(104,80),P(86,76),P(72,76)],11,8)
  for (px,py) in[(112,80),(96,76)]: S.spike('orange',P(px,py-6),P(px+8,py-16),2.5)
  dx,dy=-18,22+oy
  hp=lambda pts:[(x+dx,y+dy) for x,y in pts]
  S.poly('scale',hp([(98,38),(84,34),(62,40),(40,48),(36,54),(56,58),(78,58),(96,54)]))
  S.poly('scale',hp([(92,40),(80,38),(62,44),(46,50),(60,52),(80,52)]),shift=1)
  S.poly('scaleD',hp([(88,56),(70,60),(50,62),(42,58),(60,64),(84,66)]))
  for a,b_,w in[((44,55),(38,62),2),((54,60),(50,66),1.6)]: S.spike('claw',hp([a])[0],hp([b_])[0],w)
  S.tube('orange',hp([(86,36),(98,24),(110,14),(124,10)]),3.5,1.2)
  S.tube('orange',hp([(96,26),(104,12)]),2,1,shift=1);S.tube('orange',hp([(92,40),(112,34),(126,30)]),2.2,1,shift=-1)
  S.ell('scale',*P(138,96),10,9,shift=1)
  S.tube('scale',[P(136,98),P(128,114)],7,5);S.tube('scale',[P(128,114),P(112,118)],5,4)
  S.poly('orange',[P(130,108),P(134,122),P(118,124),P(116,118)])
  for a in(-.3,0,.3): S.spike('claw',P(110,118),P(102,121+a*10),1.8)
  S.spike('orange',P(144,84),P(176,46),6);S.spike('orange',P(136,88),P(156,58),3.5,shift=-1)
  wing(S,P(148,82),th,56,84,'mem',flip=1,chaos=True)
  for a,l,w in[(-1.5,24,5),(-1.0,18,4.5),(-2.0,15,4),(-.5,12,3.5)]: S.spike('chaos',P(140,90),(P(140,90)[0]+math.cos(a)*l,P(140,90)[1]+math.sin(a)*l),w)
  S.poly('chaos',[P(118,112),P(130,116),P(122,126),P(112,122)])
  for a,l,w in[(-1.7,22,4.5),(-2.2,14,3),(-1.2,14,3)]: S.spike('chaos',hp([(86,34)])[0],(hp([(86,34)])[0][0]+math.cos(a)*l,hp([(86,34)])[0][1]+math.sin(a)*l),w)
  S.spike('chaos',P(214,126),P(226,120),3)
  S.ell('chaos',*hp([(66,44)])[0],3,1.6)
  return S

import json,base64,io
out={}
for name,fn in[('front',front),('side',side),('q3',three_q)]+[(f'fly{k}',(lambda k=k:fly(k))) for k in range(6)]+[(f'hover{k}',(lambda k=k:three_q(k))) for k in range(6)]:
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
