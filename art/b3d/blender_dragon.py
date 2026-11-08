
# 3D → 2D 像素：用 Blender（bpy，無頭模式）建出龍王 3D 模型，多角度正交算圖，再後製成像素圖
# 解決：透視（真實 3D 投影）＋一致性（同一個模型轉角度、擺動作）
# 後製：材質 ID 圖 → 每材質套固定色階；明暗圖 → 量化成階；深度圖 → 前後部件之間的內描邊；外描邊；混沌晶體 → 星空＋白紫外框＋發光
import bpy, bmesh, math, os, json, sys
from mathutils import Vector, Matrix
import numpy as np
OUT=os.path.dirname(os.path.abspath(__file__))+'/b3d'; os.makedirs(OUT,exist_ok=True)
PPU=22  # 每 1 單位 = 22 像素（所有角度共用 → 尺寸一致）

MATS=['scale','belly','mem','bone','orange','claw','chaos','eye']
ID_COL={m:((i+1)*29%256/255,(i*83+40)%256/255,(i*151+90)%256/255) for i,m in enumerate(MATS)}

def reset():
  bpy.ops.wm.read_factory_settings(use_empty=True)
  sc=bpy.context.scene;sc.render.engine='CYCLES';sc.cycles.device='CPU';sc.cycles.samples=24;sc.cycles.use_denoising=False
  sc.render.film_transparent=True;sc.render.filter_size=0.01;sc.view_settings.view_transform='Standard';sc.view_settings.look='None'
  sc.render.image_settings.file_format='PNG';sc.render.image_settings.color_mode='RGBA'
  return sc

def mat_lum(name,bump=0):
  m=bpy.data.materials.new('L_'+name);m.use_nodes=True;nt=m.node_tree;b=nt.nodes['Principled BSDF']
  b.inputs['Base Color'].default_value=(.8,.8,.8,1);b.inputs['Roughness'].default_value=1;b.inputs['Specular IOR Level'].default_value=0
  if bump:
    v=nt.nodes.new('ShaderNodeTexVoronoi');v.inputs['Scale'].default_value=bump;v.feature='DISTANCE_TO_EDGE'
    bp=nt.nodes.new('ShaderNodeBump');bp.inputs['Strength'].default_value=.6;nt.links.new(v.outputs['Distance'],bp.inputs['Height']);nt.links.new(bp.outputs['Normal'],b.inputs['Normal'])
  return m
def mat_emit(name,col):
  m=bpy.data.materials.new(name);m.use_nodes=True;nt=m.node_tree;nt.nodes.clear()
  e=nt.nodes.new('ShaderNodeEmission');e.inputs['Color'].default_value=(*col,1);o=nt.nodes.new('ShaderNodeOutputMaterial');nt.links.new(e.outputs[0],o.inputs['Surface']);return m
def mat_depth():
  m=bpy.data.materials.new('DEPTH');m.use_nodes=True;nt=m.node_tree;nt.nodes.clear()
  cd=nt.nodes.new('ShaderNodeCameraData');mt=nt.nodes.new('ShaderNodeMath');mt.operation='DIVIDE';mt.inputs[1].default_value=60
  e=nt.nodes.new('ShaderNodeEmission');o=nt.nodes.new('ShaderNodeOutputMaterial')
  nt.links.new(cd.outputs['View Z Depth'],mt.inputs[0]);nt.links.new(mt.outputs[0],e.inputs['Color']);nt.links.new(e.outputs[0],o.inputs['Surface']);return m

PARTS=[]  # (obj, [mat per slot])
def add_obj(name,me,mats):
  o=bpy.data.objects.new(name,me);bpy.context.scene.collection.objects.link(o);PARTS.append((o,mats));return o

def skin(name,verts,edges,radii,mat,sub=2):
  me=bpy.data.meshes.new(name);me.from_pydata([tuple(v) for v in verts],edges,[]);me.update()
  o=add_obj(name,me,[mat]);sk=o.modifiers.new('skin','SKIN');sk.use_smooth_shade=True
  for i,r in enumerate(radii):me.skin_vertices[0].data[i].radius=(r,r)
  me.skin_vertices[0].data[0].use_root=True
  s=o.modifiers.new('sub','SUBSURF');s.levels=sub;s.render_levels=sub;return o

def cone(name,base,tip,r,mat,seg=6):
  base=Vector(base);tip=Vector(tip);d=tip-base;L=d.length
  bm=bmesh.new();bmesh.ops.create_cone(bm,cap_ends=True,segments=seg,radius1=r,radius2=0,depth=L)
  me=bpy.data.meshes.new(name);bm.to_mesh(me);bm.free();o=add_obj(name,me,[mat])
  o.location=base+d/2;o.rotation_mode='QUATERNION';o.rotation_quaternion=Vector((0,0,1)).rotation_difference(d.normalized());return o

def crystal(name,base,dirv,L,r):
  base=Vector(base);d=Vector(dirv).normalized()
  verts=[(0,0,0),(r,0,L*.45),(0,r,L*.45),(-r,0,L*.45),(0,-r,L*.45),(0,0,L)]
  faces=[(0,1,2),(0,2,3),(0,3,4),(0,4,1),(5,2,1),(5,3,2),(5,4,3),(5,1,4)]
  me=bpy.data.meshes.new(name);me.from_pydata(verts,[],faces);me.update();o=add_obj(name,me,['chaos'])
  o.location=base;o.rotation_mode='QUATERNION';o.rotation_quaternion=Vector((0,0,1)).rotation_difference(d);return o

def ellip(name,loc,scale,mat,rot=(0,0,0)):
  bm=bmesh.new();bmesh.ops.create_uvsphere(bm,u_segments=12,v_segments=8,radius=1);me=bpy.data.meshes.new(name);bm.to_mesh(me);bm.free()
  for p in me.polygons:p.use_smooth=True
  o=add_obj(name,me,[mat]);o.location=loc;o.scale=scale;o.rotation_euler=rot;return o

def wing(name,root,side,flap,sweep,chaos=False):
  # 局部座標：s＝往外（±Y），b＝往後（+X），u＝往上（+Z）；整片翅膀以身體軸（X）為軸旋轉 flap
  pts={'root':(0,0,0),'wrist':(2.6,-.3,.8),'tip':(5.4,.4,.6),'f1':(5.0,2.2,-.2),'f2':(3.9,3.4,-.5),'f3':(2.6,4.0,-.6),'f4':(1.2,3.5,-.4),'back':(.3,2.7,-.2)}
  R=Matrix.Rotation(math.radians(flap)*side,4,'X')@Matrix.Rotation(math.radians(sweep)*side,4,'Z')
  def W(p):
    s,b,u=p;v=Vector((b,s*side,u));return Vector(root)+(R@v)
  P={k:W(v) for k,v in pts.items()}
  def mid(a,b,t=.5,pull=.18):
    m=P[a].lerp(P[b],t);return m+(P['wrist']-m)*pull
  ring=['tip','f1','f2','f3','f4','back']
  verts=[P['wrist'],P['root']];faces=[];mats=[]
  vid={}
  for k in ring:vid[k]=len(verts);verts.append(P[k])
  for a,b in zip(ring,ring[1:]):
    mi=len(verts);verts.append(mid(a,b))
    faces+= [(0,vid[a],mi),(0,mi,vid[b])]
  faces.append((0,vid['back'],1))
  me=bpy.data.meshes.new(name);me.from_pydata([tuple(v) for v in verts],[],faces);me.update()
  slots=['mem','chaos'] if chaos else ['mem']
  o=add_obj(name,me,slots)
  if chaos:
    for i,p in enumerate(me.polygons):
      if i in (0,1,2):p.material_index=1  # 翼尖附近的翼膜被侵蝕成星空
  so=o.modifiers.new('sol','SOLIDIFY');so.thickness=.06
  # 翼骨
  bv=[P['root'],P['wrist'],P['tip'],P['f1'],P['f2'],P['f3'],P['f4']];be=[(0,1),(1,2),(1,3),(1,4),(1,5),(1,6)]
  skin(name+'_bone',bv,be,[.26,.2,.08,.08,.08,.08,.08],'bone',sub=1)
  cone(name+'_thumb',P['wrist'],P['wrist']+Vector((-.3,0,.7)),.12,'claw')
  if chaos:
    for k,dv,L in [('tip',(0,-.3,1),1.1),('f1',(.3,-.5,.6),.9),('wrist',(-.2,-.6,1),.8)]:
      crystal(name+'_cx'+k,P['wrist'].lerp(P[k],.7),dv,L*1.3,.22)
  return P

def build(pose):
  global PARTS;PARTS=[]
  if pose=='stand':
    sp=[(6.2,0,.25),(4.8,0,.5),(3.4,0,.9),(2.0,0,1.9),(1.0,0,3.0),(.2,0,4.1),(-.2,0,5.0),(-.7,0,5.8),(-1.2,0,6.35),(-2.5,0,6.15)]
    legs=lambda s:[(1.9,.75*s,2.0),(.9,.95*s,1.35),(1.6,.95*s,.55),(.9,.95*s,.08),(.4,.95*s,0)]
    arms=lambda s:[(.0,.85*s,4.2),(-.6,1.35*s,3.3),(-1.25,1.25*s,2.7)]
    wroot=(.4,.55,4.8);flap=38;sweep=-15
  else:
    sp=[(8.2,0,4.6),(6.6,0,4.85),(5.0,0,5.0),(3.4,0,5.0),(2.0,0,4.95),(.7,0,4.95),(-.6,0,5.05),(-1.6,0,5.3),(-2.6,0,5.6),(-3.9,0,5.45)]
    legs=lambda s:[(2.4,.7*s,4.8),(2.9,.85*s,4.1),(3.7,.85*s,3.95),(4.4,.85*s,4.05),(4.8,.85*s,4.1)]
    arms=lambda s:[(-.5,.75*s,4.75),(-1.0,1.0*s,4.15),(-1.8,.85*s,4.0)]
    wroot=(-.2,.55,5.65);flap=pose;sweep=-10
  rad=[.07,.22,.42,.85,1.0,.95,.6,.45,.58,.3]
  skin('body',sp,[(i,i+1) for i in range(len(sp)-1)],rad,'scale')
  head=Vector(sp[8]);snout=Vector(sp[9])
  # 頭：下顎、眉骨
  ellip('jaw',head.lerp(snout,.55)+Vector((0,0,-.28)),(.75,.32,.16),'scale')
  for s in(1,-1):
    cone('brow%d'%s,head+Vector((-.1,.3*s,.35)),head+Vector((.5,.55*s,.75)),.13,'scale')
    ellip('eye%d'%s,head+Vector((-.42,.33*s,.18)),(.1,.06,.06),'eye')
    # 豎琴形角冠
    a=head+Vector((.15,.25*s,.45));b=a+Vector((.55,.25*s,.9));c=b+Vector((.5,.05*s,.75))
    skin('horn%d'%s,[a,b,c],[(0,1),(1,2)],[.13,.09,.03],'orange',sub=1)
    cone('hornb%d'%s,b,b+Vector((.55,.45*s,.15)),.07,'orange');cone('hornc%d'%s,a.lerp(b,.4),a.lerp(b,.4)+Vector((.6,.1*s,-.15)),.06,'orange')
  for k in range(3):cone('tooth%d'%k,snout+Vector((.25+k*.22,0,-.3)),snout+Vector((.2+k*.22,0,-.5)),.05,'claw')
  # 腹甲
  for k,t in enumerate([.35,.5,.65,.8]):
    p=Vector(sp[3]).lerp(Vector(sp[6]),t);n=(Vector(sp[6])-Vector(sp[3])).normalized()
    fwd=Vector((-n.z,0,n.x)) if pose=='stand' else Vector((0,0,-1))
    if pose=='stand':fwd=Vector((-1,0,.15)).normalized()
    ellip('belly%d'%k,p+fwd*.78,(.32,.55,.42) if pose=='stand' else (.45,.55,.28),'belly')
  # 背刺
  for k in range(1,8):
    p=Vector(sp[k]);q=Vector(sp[k+1]) if k<9 else p;up=Vector((0,0,1)) if pose!='stand' or k<3 else Vector((.6,0,.8)).normalized()
    cone('bs%d'%k,p+up*rad[k]*.85,p+up*(rad[k]+.45)+(p-q).normalized()*.25,.12,'orange')
  for side in(1,-1):
    L=legs(side);skin('leg%d'%side,L,[(0,1),(1,2),(2,3),(3,4)],[.55,.4,.3,.22,.14],'scale')
    cone('kneesp%d'%side,Vector(L[1]),Vector(L[1])+Vector((-.5,0,.15)) if pose=='stand' else Vector(L[1])+Vector((.1,0,-.45)),.13,'orange')
    for t in(-.25,0,.25):cone('toe%d%.2f'%(side,t),Vector(L[4]),Vector(L[4])+Vector((-.4,t*side,-.08)),.07,'claw')
    A=arms(side);skin('arm%d'%side,A,[(0,1),(1,2)],[.42,.3,.22],'scale')
    e=Vector(A[1]);h=Vector(A[2]);cone('elbowblade%d'%side,e+Vector((.1,0,.05)),e+(e-Vector(A[0])).normalized()*.1+Vector((.9,.25*side,-.35)),.18,'orange')
    for t in(-.3,0,.3):cone('claw%d%.1f'%(side,t),h,h+Vector((-.4,t*side*.6,-.35)),.06,'claw')
    sh=Vector(A[0])
    cone('shoulderspike%d'%side,sh+Vector((.2,.1*side,.3)),sh+(Vector((.9,.55*side,2.6)) if pose=='stand' else Vector((1.8,.5*side,1.2))),.3,'orange')
    P=wing('wing%d'%side,(wroot[0],wroot[1]*side,wroot[2]),side,flap,sweep,chaos=(side==-1))
  # 混沌晶體（-Y 側＝被侵蝕的那一側）
  sh=Vector(arms(-1)[0]);crystal('cx_sh0',sh+Vector((0,-.25,.25)),(0,-.6,1),1.9,.32);crystal('cx_sh1',sh+Vector((.2,-.3,.1)),(.3,-1,.5),1.5,.26);crystal('cx_sh2',sh+Vector((-.2,-.2,.2)),(-.4,-.6,.8),1.3,.24);crystal('cx_sh3',sh+Vector((.3,-.3,.3)),(.6,-.4,1),1.2,.22)
  e=Vector(arms(-1)[1]);crystal('cx_arm0',e+Vector((0,-.2,0)),(0,-1,.3),1.1,.2);crystal('cx_arm1',e+Vector((-.2,-.2,-.2)),(-.3,-1,-.3),.9,.18)
  crystal('cx_head',head+Vector((.05,0,.5)),(.1,0,1),1.4,.22);crystal('cx_head2',head+Vector((.15,-.15,.45)),(.3,-.5,1),.7,.12)
  chest=Vector(sp[5]);fw=Vector((-1,0,.1)).normalized() if pose=='stand' else Vector((0,0,-1))
  c=crystal('cx_core',chest+fw*.9-fw*.0,fw,.5,.3)
  L=legs(-1);crystal('cx_leg',Vector(L[2])+Vector((0,-.2,0)),(.2,-1,.4),.6,.12)

ROLE_BUMP={'scale':14,'belly':0,'mem':0,'bone':0,'orange':0,'claw':0,'chaos':0,'eye':0}
def setup_mats(passname):
  cache={}
  for o,slots in PARTS:
    for si,role in enumerate(slots):
      key=(passname,role)
      if key not in cache:
        if passname=='id':cache[key]=mat_emit('ID_'+role,ID_COL[role])
        elif passname=='depth':cache[key]=DEPTHM
        else:cache[key]=mat_lum(role,ROLE_BUMP[role]) if role not in('chaos','eye') else mat_emit('LE_'+role,(1,1,1))
      if len(o.data.materials)>si:o.data.materials[si]=cache[key]
      else:o.data.materials.append(cache[key])

def camera(view,center):
  cam=bpy.data.cameras.new('cam');cam.type='ORTHO';co=bpy.data.objects.new('cam',cam);bpy.context.scene.collection.objects.link(co);bpy.context.scene.camera=co
  d={'front':(-1,0,0),'side':(0,-1,0),'q3':(-.72,-.69,0)}[view];el=math.radians({'front':8,'side':4,'q3':12}[view])
  d=Vector(d).normalized();d=(d*math.cos(el)+Vector((0,0,math.sin(el)))).normalized()
  co.location=Vector(center)+d*40;co.rotation_mode='QUATERNION';co.rotation_quaternion=(-d).to_track_quat('-Z','Y')
  return co,d

def light(d):
  # 光源固定在畫面左上前方（和遊戲其他美術一致）
  sun=bpy.data.lights.new('sun','SUN');sun.energy=2.3;sun.angle=math.radians(2);so=bpy.data.objects.new('sun',sun);bpy.context.scene.collection.objects.link(so)
  cam=bpy.context.scene.camera;q=cam.rotation_quaternion;right=q@Vector((1,0,0));up=q@Vector((0,1,0))
  ld=(-right*.8+up*1.0+d*.9).normalized();so.rotation_mode='QUATERNION';so.rotation_quaternion=(-ld).to_track_quat('-Z','Y')
  w=bpy.data.worlds.new('w');w.use_nodes=True;w.node_tree.nodes['Background'].inputs['Color'].default_value=(.09,.09,.12,1);bpy.context.scene.world=w

def render(view,center,W,H,tag):
  sc=bpy.context.scene;co,d=camera(view,center);co.data.ortho_scale=max(W,H)/PPU;sc.render.resolution_x=W;sc.render.resolution_y=H;sc.render.resolution_percentage=100
  light(d);res={}
  for pn in('id','depth','lum'):
    setup_mats(pn);sc.cycles.samples=1 if pn!='lum' else 48;sc.world.node_tree.nodes['Background'].inputs['Strength'].default_value=0 if pn!='lum' else 1
    for o in sc.objects:
      if o.type=='LIGHT':o.hide_render=(pn!='lum')
    fp=f'{OUT}/{tag}_{pn}.png';sc.render.filepath=fp;bpy.ops.render.render(write_still=True);res[pn]=fp
  # 胸口（錨點）在畫面上的位置
  from bpy_extras.object_utils import world_to_camera_view
  a=world_to_camera_view(sc,co,Vector(ANCHOR));res['anchor']=(a.x*W,(1-a.y)*H)
  return res

if __name__=='__main__':
  jobs=[('stand','front',(0,0,3.9),300,260,'front'),('stand','q3',(0,0,3.9),300,260,'q3'),('stand','side',(1.2,0,3.6),340,260,'side')]
  for k in range(6):
    f=-38+108*(1-math.cos(k/6*2*math.pi))/2  # 振翅角度：-38°（下拍）↔ 70°（上舉）
    jobs.append((70-108*(1-math.cos(k/6*2*math.pi))/2 ,'side',(1.6,0,5.0),360,280,f'fly{k}'))
  meta={}
  for pose,view,center,W,H,tag in jobs:
    sc=reset();DEPTHM=mat_depth();globals()['DEPTHM']=DEPTHM
    build(pose);ANCHOR=(0,0,4.1) if pose=='stand' else (.3,0,5.0);globals()['ANCHOR']=ANCHOR
    r=render(view,center,W,H,tag);meta[tag]=r;print('rendered',tag,flush=True)
  json.dump(meta,open(f'{OUT}/meta.json','w'))
  sc=reset();DEPTHM=mat_depth();globals()['DEPTHM']=DEPTHM;build('stand');setup_mats('lum');bpy.ops.wm.save_as_mainfile(filepath=f'{OUT}/dragon_king.blend')
