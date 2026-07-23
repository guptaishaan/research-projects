"""
Version B data generator: fine-tune CLIP on each domain (ORIGINAL recipe: AdamW, image-encoder
+ linear head, text frozen) at 3 learning rates, and LOG THE LEARNING DYNAMICS every epoch.

Per (domain, lr, epoch) we record:
  forgetting   : mean zero-shot accuracy drop on generalist benchmarks (the TARGET curve)
  train_loss   : average training loss this epoch
  train_acc    : average training accuracy this epoch
  param_drift  : ||visual_params_now - visual_params_pristine||_2       (how far the encoder moved)
  emb_drift    : mean L2 shift of image embeddings on a fixed probe set  (how much the representation moved)

Goal downstream: from cheap STATIC features + the FIRST FEW epochs of these dynamics, forecast the
FINAL forgetting -> an "early-warning" system. Everything checkpoints incrementally (atomic).

Also writes static per-domain features (semantic / embed / spectral / frechet distance) to features.csv.
"""
import os, copy, time, json, argparse, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision as tv
from torch.utils.data import DataLoader, Subset
import open_clip
from scipy.linalg import sqrtm

SEED=42; np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEV="cuda"; ROOT="data"; OUT="outputs/dynamics"; os.makedirs(OUT, exist_ok=True)

ap=argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--train-n", type=int, default=4000)
ap.add_argument("--eval-n", type=int, default=1500)
ap.add_argument("--lrs", type=float, nargs="+", default=[1e-6,5e-6,1e-5])
ap.add_argument("--smoke", nargs="*", default=None)
args=ap.parse_args()

model,_,preprocess=open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
tokenizer=open_clip.get_tokenizer("ViT-B-32"); model=model.to(DEV).eval()
PRISTINE=copy.deepcopy(model.state_dict())
def reset(): model.load_state_dict(PRISTINE); model.eval()
EMB=model.visual.output_dim

def prompted(cs): return [f"a photo of a {c.replace('_',' ')}." for c in cs]
GTSRB_NAMES=["speed limit 20","speed limit 30","speed limit 50","speed limit 60","speed limit 70",
 "speed limit 80","end of speed limit 80","speed limit 100","speed limit 120","no passing",
 "no passing for trucks","right of way at intersection","priority road","yield","stop","no vehicles",
 "no trucks","no entry","general caution","dangerous curve left","dangerous curve right","double curve",
 "bumpy road","slippery road","road narrows on right","road work","traffic signals","pedestrians",
 "children crossing","bicycles crossing","beware of ice or snow","wild animals crossing",
 "end of all speed and passing limits","turn right ahead","turn left ahead","ahead only",
 "go straight or right","go straight or left","keep right","keep left","roundabout mandatory",
 "end of no passing","end of no passing for trucks"]
DIGIT_NAMES=[f"the digit {i}" for i in range(10)]
FAMILY={"OxfordPets":"natural","Food101":"natural","FGVCAircraft":"natural","DTD":"natural","EuroSAT":"natural",
 "SVHN":"digits","MNIST":"digits","GTSRB":"digits",
 "pathmnist":"medical","pneumoniamnist":"medical","bloodmnist":"medical","dermamnist":"medical",
 "octmnist":"medical","retinamnist":"medical","breastmnist":"medical","organcmnist":"medical"}

class MedWrap(torch.utils.data.Dataset):
    def __init__(s,ds): s.ds=ds
    def __len__(s): return len(s.ds)
    def __getitem__(s,i):
        img,y=s.ds[i]
        if img.mode!="RGB": img=img.convert("RGB")
        return preprocess(img), int(np.array(y).ravel()[0])
class LabelMap(torch.utils.data.Dataset):
    def __init__(s,ds): s.ds=ds
    def __len__(s): return len(s.ds)
    def __getitem__(s,i):
        img,y=s.ds[i]; return preprocess(img if img.mode=="RGB" else img.convert("RGB")), int(y)

def build_domains():
    S={}
    euro=tv.datasets.EuroSAT(ROOT,transform=preprocess); S["EuroSAT"]=(euro,euro.classes)
    gtsrb=tv.datasets.GTSRB(ROOT,split="train",transform=preprocess); S["GTSRB"]=(gtsrb,GTSRB_NAMES)
    air=tv.datasets.FGVCAircraft(ROOT,split="trainval",transform=preprocess); S["FGVCAircraft"]=(air,air.classes)
    svhn=tv.datasets.SVHN(ROOT,split="train",transform=preprocess); S["SVHN"]=(svhn,DIGIT_NAMES)
    dtd=tv.datasets.DTD(ROOT,split="train",transform=preprocess); S["DTD"]=(dtd,dtd.classes)
    mnist=tv.datasets.MNIST(ROOT,train=True); S["MNIST"]=(LabelMap(mnist),DIGIT_NAMES)
    for name,fn,split in [("OxfordPets",tv.datasets.OxfordIIITPet,"trainval"),("Food101",tv.datasets.Food101,"train")]:
        try:
            d=fn(ROOT,split=split,transform=preprocess); S[name]=(d,d.classes)
        except Exception as e: print(f"skip {name}: {e}",flush=True)
    import medmnist; from medmnist import INFO
    for flag in ["pathmnist","pneumoniamnist","bloodmnist","dermamnist","octmnist","retinamnist","breastmnist","organcmnist"]:
        DS=getattr(medmnist,INFO[flag]["python_class"])
        S[flag]=(MedWrap(DS(split="train",download=True,root=ROOT)),list(INFO[flag]["label"].values()))
    return S

def build_benchmarks():
    return {"CIFAR100":(tv.datasets.CIFAR100(ROOT,train=False,transform=preprocess), tv.datasets.CIFAR100(ROOT,train=False).classes),
            "CIFAR10":(tv.datasets.CIFAR10(ROOT,train=False,transform=preprocess), tv.datasets.CIFAR10(ROOT,train=False).classes),
            "STL10":(tv.datasets.STL10(ROOT,split="test",transform=preprocess), tv.datasets.STL10(ROOT,split="test").classes)}

def subset(ds,n):
    if n is None or n>=len(ds): return ds
    idx=np.random.RandomState(SEED).choice(len(ds),n,replace=False); return Subset(ds,idx.tolist())

@torch.no_grad()
def zeroshot_acc(ds, classnames, n):
    dl=DataLoader(subset(ds,n), batch_size=256, num_workers=4)
    W=F.normalize(model.encode_text(tokenizer(prompted(classnames)).to(DEV)), dim=-1)
    c=t=0; model.eval()
    for x,y in dl:
        f=F.normalize(model.encode_image(x.to(DEV)), dim=-1)
        c+=((100.*f@W.T).argmax(1).cpu()==torch.as_tensor(y)).sum().item(); t+=len(y)
    return c/t

def all_bench(BENCH,n): return {k:zeroshot_acc(ds,cls,n) for k,(ds,cls) in BENCH.items()}

@torch.no_grad()
def probe_embeddings(probe_x):
    model.eval(); return F.normalize(model.encode_image(probe_x), dim=-1)

def visual_params_vec():
    return torch.cat([p.detach().flatten() for p in model.visual.parameters()])

# ---------------- static cheap features ----------------
@torch.no_grad()
def mean_img_emb(ds,n=1000):
    dl=DataLoader(subset(ds,n),batch_size=256,num_workers=4); fs=[]
    for x,_ in dl: fs.append(F.normalize(model.encode_image(x.to(DEV)),dim=-1).cpu())
    return torch.cat(fs).numpy()
def radial_spectrum(ds,n=1000,size=64):
    dl=DataLoader(subset(ds,n),batch_size=256,num_workers=4); acc=None;cnt=0
    for x,_ in dl:
        g=x.mean(1,keepdim=True); g=F.interpolate(g,size=(size,size),mode="bilinear",align_corners=False).squeeze(1).numpy()
        P=np.abs(np.fft.fftshift(np.fft.fft2(g),axes=(1,2)))**2
        s=P.sum(0); acc=s if acc is None else acc+s; cnt+=len(g)
    P=acc/cnt; c=size//2; yy,xx=np.mgrid[-c:c,-c:c]; RAD=np.sqrt(xx**2+yy**2)
    prof=np.array([P[(RAD>=k)&(RAD<k+1)].mean() for k in range(c)]); return prof/prof.sum()
def frechet(mu1,cov1,mu2,cov2):
    d=mu1-mu2; cc,_=sqrtm(cov1@cov2,disp=False); cc=cc.real
    return float(d@d + np.trace(cov1+cov2-2*cc))

def compute_static_features(S, BENCH):
    reset()
    ref_emb=mean_img_emb(BENCH["CIFAR100"][0]); ref_mean=ref_emb.mean(0)
    ref_mu,ref_cov=ref_emb.mean(0),np.cov(ref_emb,rowvar=False)
    ref_spec=radial_spectrum(BENCH["CIFAR100"][0])
    rows=[]
    for name,(ds,cls) in S.items():
        emb=mean_img_emb(ds); acc=zeroshot_acc(ds,cls,1000)
        mu,cov=emb.mean(0),np.cov(emb,rowvar=False)
        rows.append(dict(domain=name, family=FAMILY.get(name,"?"),
            semantic_dist=float(1-acc),
            clip_embed_dist=float(np.linalg.norm(emb.mean(0)-ref_mean)),
            spectral_dist=float(np.linalg.norm(radial_spectrum(ds)-ref_spec)),
            frechet_dist=frechet(mu,cov,ref_mu,ref_cov)))
        print("feat", name, "sem=%.3f"%rows[-1]["semantic_dist"], flush=True)
    pd.DataFrame(rows).set_index("domain").to_csv(f"{OUT}/features.csv")

# ---------------- training with per-epoch dynamics logging ----------------
def atomic(df,path): tmp=path+".tmp"; df.to_csv(tmp,index=False); os.replace(tmp,path)

def run():
    BENCH=build_benchmarks()
    # fixed probe set for embedding-drift (500 CIFAR100 test imgs)
    probe_ds=subset(BENCH["CIFAR100"][0],500)
    probe_x=torch.stack([probe_ds[i][0] for i in range(len(probe_ds))]).to(DEV)
    reset(); base=all_bench(BENCH,args.eval_n); pristine_probe=probe_embeddings(probe_x); theta0=visual_params_vec()
    print("baseline zs:",{k:round(v,3) for k,v in base.items()},flush=True)
    pd.Series(base).to_csv(f"{OUT}/baseline.csv")

    S=build_domains()
    if not os.path.exists(f"{OUT}/features.csv"): compute_static_features(S,BENCH)

    curves_path=f"{OUT}/dynamics_curves.csv"
    rows=pd.read_csv(curves_path).to_dict("records") if os.path.exists(curves_path) else []
    done={(r["domain"],float(r["lr"])) for r in rows}
    todo=list(S) if not args.smoke else [d for d in args.smoke if d in S]

    for name in todo:
        ds,cls=S[name]
        for lr in args.lrs:
            if (name,float(lr)) in done: print(f"skip {name} lr={lr}",flush=True); continue
            reset(); theta0=visual_params_vec(); pristine_probe=probe_embeddings(probe_x)
            head=nn.Linear(EMB,len(cls)).to(DEV)
            for p in model.parameters(): p.requires_grad_(True)
            opt=torch.optim.AdamW([{"params":model.visual.parameters(),"lr":lr},
                                   {"params":head.parameters(),"lr":1e-3}], weight_decay=1e-4)
            dl=DataLoader(subset(ds,args.train_n),batch_size=128,shuffle=True,num_workers=4,drop_last=True)
            t0=time.time()
            for ep in range(1,args.epochs+1):
                model.train(); tl=ta=nb=ntot=0
                for x,y in dl:
                    x=x.to(DEV); y=torch.as_tensor(y).long().to(DEV)
                    logits=head(model.encode_image(x)); loss=F.cross_entropy(logits,y)
                    opt.zero_grad(); loss.backward(); opt.step()
                    tl+=loss.item()*len(y); ta+=(logits.argmax(1)==y).sum().item(); nb+=1; ntot+=len(y)
                model.eval()
                post=all_bench(BENCH,args.eval_n); forget=float(np.mean([base[k]-post[k] for k in base]))
                pdrift=float((visual_params_vec()-theta0).norm().item())
                edrift=float((probe_embeddings(probe_x)-pristine_probe).norm(dim=-1).mean().item())
                rows.append(dict(domain=name, family=FAMILY.get(name,"?"), lr=float(lr), epoch=ep,
                    forgetting=forget, train_loss=tl/ntot, train_acc=ta/ntot,
                    param_drift=pdrift, emb_drift=edrift))
                atomic(pd.DataFrame(rows), curves_path)
            print(f"[{name} lr={lr:.0e}] final_forget={forget:+.4f}  ({time.time()-t0:.0f}s)",flush=True)
    print("DONE",curves_path,flush=True)

if __name__=="__main__": run()
