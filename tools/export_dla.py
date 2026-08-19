import argparse
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


class B(nn.Module):
 def __init__(s,ci,co,st=1):
  super().__init__();s.c1=nn.Conv2d(ci,co,3,st,1,bias=False);s.n1=nn.BatchNorm2d(co);s.c2=nn.Conv2d(co,co,3,padding=1,bias=False);s.n2=nn.BatchNorm2d(co);s.d=None if st==1 and ci==co else nn.Sequential(nn.AvgPool2d(2,st,ceil_mode=True,count_include_pad=False),nn.Conv2d(ci,co,1,bias=False),nn.BatchNorm2d(co))
 def forward(s,x):
  y=s.n2(s.c2(F.relu(s.n1(s.c1(x)))));return F.relu(y+(s.d(x) if s.d else x))

class BB(nn.Module):
 def __init__(s):
  super().__init__();s.c=nn.Sequential(nn.Conv2d(3,32,3,2,1,bias=False),nn.BatchNorm2d(32),nn.ReLU(),nn.Conv2d(32,32,3,padding=1,bias=False),nn.BatchNorm2d(32),nn.ReLU(),nn.Conv2d(32,64,3,padding=1,bias=False));s.n=nn.BatchNorm2d(64);s.pe=nn.Conv2d(6,64,1);s.f=nn.Conv2d(128,64,1);s.p=nn.MaxPool2d(3,2,1);s.l1=nn.Sequential(B(64,64),B(64,64));s.l2=nn.Sequential(B(64,128,2),B(128,128));s.l3=nn.Sequential(B(128,256,2),B(256,256));s.l4=nn.Sequential(B(256,512,2),B(512,512))
 def forward(s,r,p):
  r=F.relu(s.n(s.c(r)));x=s.p(s.f(torch.cat((r,s.pe(p)),1)));a=s.l1(x);b=s.l2(a);c=s.l3(b);d=s.l4(c);return a,b,c,d

class FM(nn.Module):
 def __init__(s):
  super().__init__();s.gs=torch.tensor([10.,10.,3.14159265,1.,1.,1.]);s.fg=nn.Conv2d(9,512,1);s.fb=nn.Conv2d(9,512,1);s.f4=nn.Conv2d(512,256,1);s.f3=nn.Conv2d(256,256,1);s.st=nn.Conv2d(9,256,1);s.ad=nn.Conv2d(256,256,1);s.w=nn.Conv2d(256,256,1);s.de=nn.Conv2d(256,256,1);s.hp=nn.Conv2d(256,256,1);s.hs=nn.Conv2d(256,256,1,bias=False);s.hp.weight.requires_grad_(False);s.hp.bias.requires_grad_(False);s.hs.weight.requires_grad_(False);s.hp.weight.zero_();s.hs.weight.zero_();s.hp.weight[:,:,0,0].copy_(torch.eye(256));s.hp.bias.fill_(3);s.hs.weight[:,:,0,0].copy_(torch.eye(256)/6)
 def forward(s,f4,f3,t,g):
  z=torch.cat((g,t),1);q=f4+f4*s.fg(z)+s.fb(z);x=F.relu(F.interpolate(s.f4(q),size=f3.shape[-2:],mode="nearest")+s.f3(f3));a=F.relu(s.ad(x)+s.st(z));return x+s.hs(F.relu6(s.hp(s.w(a))))*s.de(x)

class C(nn.Module):
 def __init__(s):
  super().__init__();s.a=nn.Conv2d(256,256,3,2,1);s.b=nn.Conv2d(256,256,3,2,1);s.c=nn.Conv2d(256,256,(6,8))
 def forward(s,x):return s.c(F.relu(s.b(F.relu(s.a(x)))))

class SRU(nn.Module):
 def __init__(s):
  super().__init__();s.ai=nn.Conv2d(524,256,1);s.af=nn.Conv2d(524,256,1);s.ao=nn.Conv2d(524,256,1);s.ag=nn.Conv2d(524,256,1);s.t=nn.Conv2d(268,256,1);s.neg=nn.Conv2d(256,256,1,bias=False);s.ident=nn.Conv2d(256,256,1,bias=False);s.neg.weight.requires_grad_(False);s.ident.weight.requires_grad_(False);s.neg.weight.zero_();s.ident.weight.zero_();s.neg.weight[:, :, 0, 0].copy_(-torch.eye(256));s.ident.weight[:, :, 0, 0].copy_(torch.eye(256))
 def forward(s,x,h,c):
  q=torch.cat((x,h),1);i,f,o,g=s.ai(q),s.af(q),s.ao(q),s.ag(q);i,f,o=torch.sigmoid(i),torch.sigmoid(f),torch.sigmoid(o);g=torch.tanh(s.t(x)*g);ff=f*s.ident(f);a=i*f;bb=s.neg(i*ff);f=a+s.ident(a)+bb+s.ident(bb)+ff;c=f*c+g-f*g;return o*torch.tanh(c),c

class TF(nn.Module):
 def __init__(s):
  super().__init__();s.gs=torch.tensor([10.,10.,3.14159265,1.,1.,1.]);s.ds=torch.tensor([.1,.1,.1]);s.c=C();s.r=SRU();s.g=nn.Conv2d(256,256,1);s.b=nn.Conv2d(256,256,1);s.h=C()
 def forward(s,x,t,d,g,h,c):
  z=torch.cat((s.c(x),t,d,g),1);h,c=s.r(z,h,c);x=x+x*s.g(h)+s.b(h);return s.h(x),h,c

class DH(nn.Module):
 def __init__(s):
  super().__init__();s.l=nn.ModuleList(nn.Conv2d(c,64,1) for c in (512,256,128,64));s.r=nn.ModuleList(nn.Conv2d(64,64,3,padding=1) for _ in range(4));s.p=nn.ModuleList(nn.Conv2d(64,1,1) for _ in range(4))
 def forward(s,f4,f3,f2,f1):
  x=s.l[0](f4);o=[]
  for i,f in enumerate((f4,f3,f2,f1)):
   if i:x=s.l[i](f)+F.interpolate(x,size=f.shape[-2:],mode="nearest")
   o.append(torch.sigmoid(s.p[i](F.relu(s.r[i](x)))))
  return o

class TH(nn.Module):
 def __init__(s):
  super().__init__();s.c=nn.Conv2d(512,256,1);s.t=nn.Conv2d(256,1280,1);s.a=nn.Conv2d(64,64,(1,3),padding=(0,1));s.b=nn.Conv2d(64,64,(1,3),padding=(0,1));s.o=nn.Conv2d(64,7,1)
 def forward(s,x,h):
  x=s.t(F.relu(s.c(torch.cat((x,h),1))));x=x.reshape(-1,64,1,20);return s.o(x+s.b(F.relu(s.a(x))))

class M(nn.Module):
 def __init__(s):
  super().__init__();s.backbone=BB();s.fm=FM();s.temporal=TF();s.depth=DH();s.traj=TH()
 def forward(s,r,p,t,d,g,h,c):
  f1,f2,f3,f4=s.backbone(r,p);x=s.fm(f4,f3,t,g);x,h,c=s.temporal(x,t,d,g,h,c);return (s.traj(x,h),*s.depth(f4,f3,f2,f1),h,c)

def export(checkpoint: Path, output: Path, optimize: bool = True):
 m=M().eval()
 st=torch.load(checkpoint,map_location="cpu",weights_only=True)["state_dict"]
 mp={"feature_modulation.":"fm.","temporal_fuser.":"temporal.","depth_head.":"depth.","trajectory_head.":"traj."}
 for a,b in mp.items():st={k.replace(a,b) : v for k,v in st.items()}
 rep={"fm.f4_conv_1x1":"fm.f4","fm.f3_conv_1x1":"fm.f3","fm.spatial_gate.state_linear":"fm.st","fm.spatial_gate.adjust_conv_1x1":"fm.ad","fm.spatial_gate.weight_conv_1x1":"fm.w","fm.spatial_gate.feat_delta_conv_1x1":"fm.de","fm.film_by_goal_n_twist.gamma_linear":"fm.fg","fm.film_by_goal_n_twist.beta_linear":"fm.fb","temporal.vfeat_compressor.conv1":"temporal.c.a","temporal.vfeat_compressor.conv2":"temporal.c.b","temporal.vfeat_compressor.conv3":"temporal.c.c","temporal.sru.cells.0.linear_all":"temporal.r.a","temporal.sru.cells.0.transform_gate":"temporal.r.t","temporal.temporal_film.gamma_linear":"temporal.g","temporal.temporal_film.beta_linear":"temporal.b","temporal.history_enhanced_compressor.conv1":"temporal.h.a","temporal.history_enhanced_compressor.conv2":"temporal.h.b","temporal.history_enhanced_compressor.conv3":"temporal.h.c","depth.laterals":"depth.l","depth.refinements":"depth.r","depth.predictors":"depth.p","traj.temporal_conv1":"traj.a","traj.temporal_conv2":"traj.b","traj.out_conv":"traj.o"}
 rep.update({"backbone.conv1":"backbone.c","backbone.bn1":"backbone.n","backbone.pe_encoder":"backbone.pe","backbone.rgb_pe_fuse":"backbone.f","backbone.layer1":"backbone.l1","backbone.layer2":"backbone.l2","backbone.layer3":"backbone.l3","backbone.layer4":"backbone.l4"})
 for i in range(5):
  rep.update({f"backbone.l{i}.0.conv1":f"backbone.l{i}.0.c1",f"backbone.l{i}.0.bn1":f"backbone.l{i}.0.n1",f"backbone.l{i}.0.conv2":f"backbone.l{i}.0.c2",f"backbone.l{i}.0.bn2":f"backbone.l{i}.0.n2",f"backbone.l{i}.0.downsample":f"backbone.l{i}.0.d",f"backbone.l{i}.1.conv1":f"backbone.l{i}.1.c1",f"backbone.l{i}.1.bn1":f"backbone.l{i}.1.n1",f"backbone.l{i}.1.conv2":f"backbone.l{i}.1.c2",f"backbone.l{i}.1.bn2":f"backbone.l{i}.1.n2"})
 rep.update({"traj.compress":"traj.c","traj.to_seq":"traj.t"})
 st={k:v for k,v in st.items() if not k.endswith("num_batches_tracked")}
 for a,b in rep.items():st={k.replace(a,b):v for k,v in st.items()}
 wa=st.pop("temporal.r.a.weight");ba=st.pop("temporal.r.a.bias")
 for name,w,b in zip(("ai","af","ao","ag"),wa.chunk(4),ba.chunk(4)):st[f"temporal.r.{name}.weight"]=w;st[f"temporal.r.{name}.bias"]=b
 target=m.state_dict();st={k:(v[:,:,None,None] if k in target and v.ndim==2 and target[k].ndim==4 else v) for k,v in st.items()}
 missing,_=m.load_state_dict(st,strict=False)
 allowed={"fm.hp.weight","fm.hp.bias","fm.hs.weight","temporal.r.neg.weight","temporal.r.ident.weight"}
 missing=set(missing)-allowed
 if missing:
  raise RuntimeError(f"unexpected missing learned weights: {sorted(missing)}")
 z=(torch.randn(1,3,384,512),torch.randn(1,6,192,256),torch.randn(1,3,1,1),torch.randn(1,3,1,1),torch.randn(1,6,1,1),torch.zeros(1,256,1,1),torch.zeros(1,256,1,1))
 output.parent.mkdir(parents=True,exist_ok=True)
 with torch.inference_mode(): print("outputs",[tuple(x.shape) for x in m(*z)])
 raw=output.with_suffix(".raw.onnx") if optimize else output
 torch.onnx.export(m,z,str(raw),input_names=["rgb","pe","twist","delta","goal","h","c"],output_names=["trajectory","depth_f4","depth_f3","depth_f2","depth_f1","h_out","c_out"],opset_version=17,do_constant_folding=True)
 if optimize:
  import onnxruntime as ort
  options=ort.SessionOptions()
  options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
  options.optimized_model_filepath=str(output)
  ort.InferenceSession(str(raw),options,providers=["CPUExecutionProvider"])
  raw.unlink()
 print(f"exported {output}")


def main():
 parser=argparse.ArgumentParser(description="Export the RoboNav model as a DLA-safe ONNX graph.")
 parser.add_argument("checkpoint",type=Path)
 parser.add_argument("output",type=Path)
 parser.add_argument("--no-optimize",action="store_true",help="keep the raw ONNX graph")
 args=parser.parse_args()
 export(args.checkpoint,args.output,not args.no_optimize)


if __name__=="__main__":
 main()
