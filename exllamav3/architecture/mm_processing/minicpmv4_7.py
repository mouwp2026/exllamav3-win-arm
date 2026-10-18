from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _ensure_divide(length: int, divisor: int) -> int:
    return max(round(length / divisor) * divisor, divisor)


@dataclass
class MiniCPMV4_7ImageBatch:
    # NaViT patch stream: [1, C, patch, total_patches * patch]
    pixel_values: torch.Tensor
    target_sizes: torch.Tensor  # [num crops, (height, width)] in patch units
    grids: torch.Tensor  # one (rows, cols) per image; zero means unsliced
    num_patches_per_image: torch.Tensor
    input_ids: torch.Tensor | None = None
    mm_token_type_ids: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None


class MiniCPMV4_7ImageProcessor:
    """MiniCPM-V-4.7 processor, matching HF 5.18's MiniCPMV4_6 algorithm."""
    def __init__(self, scale_resolution=448, patch_size=14, max_slice_nums=9,
                 slice_mode=True, downsample_mode="16x", image_mean=(0.5,)*3,
                 image_std=(0.5,)*3, resample=Image.Resampling.BICUBIC):
        self.scale_resolution = scale_resolution
        self.patch_size = patch_size
        self.max_slice_nums = max_slice_nums
        self.slice_mode = slice_mode
        self.downsample_mode = downsample_mode
        self.merge_factor = 4 if downsample_mode == "16x" else 2
        self.image_mean = np.asarray(image_mean, dtype=np.float32).reshape(1,1,3)
        self.image_std = np.asarray(image_std, dtype=np.float32).reshape(1,1,3)
        self.resample = resample

    def find_best_resize(self, size, allow_upscale=False):
        h,w = size
        if h*w > self.scale_resolution**2 or allow_upscale:
            ar = w/h
            h = int(self.scale_resolution / math.sqrt(ar))
            w = int(h*ar)
        return (_ensure_divide(h, self.patch_size*self.merge_factor),
                _ensure_divide(w, self.patch_size*self.merge_factor))

    def get_sliced_grid(self, size):
        h,w=size
        ratio=math.ceil(h*w/self.scale_resolution**2)
        multiple=min(ratio,self.max_slice_nums)
        if multiple <= 1: return None
        best=[1,1]; err_best=float('inf'); log_ratio=math.log(w/h)
        for n in (multiple-1,multiple,multiple+1):
            if n==1 or n>self.max_slice_nums: continue
            for rows in range(1,n+1):
                if n%rows: continue
                cols=n//rows; err=abs(log_ratio-math.log(cols/rows))
                if err<err_best or (err==err_best and rows>best[0]):
                    best=[rows,cols]; err_best=err
        return best

    def get_refine_size(self, size, grid):
        h,w=size; gy,gx=grid
        rw=_ensure_divide(w,gx); rh=_ensure_divide(h,gy)
        sh,sw=self.find_best_resize((rh/gy,rw/gx),allow_upscale=True)
        return sh*gy,sw*gx

    @staticmethod
    def _patchify(x, p):
        # Same as HF reshape_by_patch, maintaining x/y patch element order.
        c,h,w=x.shape
        patches=F.unfold(x.unsqueeze(0),(p,p),stride=(p,p)).reshape(c,p,p,-1)
        return patches.permute(0,1,3,2).reshape(c,p,-1)

    def __call__(self, images: Image.Image | Sequence[Image.Image]):
        if isinstance(images, Image.Image): images=[images]
        pv=[]; target=[]; grids=[]; counts=[]
        for src in images:
            if not isinstance(src, Image.Image): src=Image.fromarray(np.asarray(src))
            src=src.convert('RGB'); w,h=src.size; size=(h,w)
            grid=self.get_sliced_grid(size) if self.slice_mode else None
            sh,sw=self.find_best_resize(size,allow_upscale=grid is None)
            thumb=src.resize((sw,sh),self.resample)
            crops=[thumb]
            slice_hw=None
            if grid:
                rh,rw=self.get_refine_size(size,grid)
                refined=src.resize((rw,rh),self.resample); gy,gx=grid
                slice_hw=(rh//gy,rw//gx)
                for y in range(gy):
                    for x in range(gx): crops.append(refined.crop((x*slice_hw[1],y*slice_hw[0],(x+1)*slice_hw[1],(y+1)*slice_hw[0])))
            for i,crop in enumerate(crops):
                arr=np.asarray(crop,dtype=np.float32)/255.0
                arr=(arr-self.image_mean)/self.image_std
                chw=torch.from_numpy(arr.transpose(2,0,1).copy())
                pv.append(self._patchify(chw,self.patch_size))
                if i==0: target.append([sh//self.patch_size,sw//self.patch_size])
                else: target.append([slice_hw[0]//self.patch_size,slice_hw[1]//self.patch_size])
            grids.append(grid if grid else [0,0]); counts.append(len(crops))
        return MiniCPMV4_7ImageBatch(torch.cat(pv,dim=-1).unsqueeze(0),
            torch.tensor(target,dtype=torch.int32),torch.tensor(grids,dtype=torch.int32),
            torch.tensor(counts,dtype=torch.int32))

    def placeholder(self, target_sizes, grid, image_token_id=248056, image_start_id=248078,
                    image_end_id=248079, slice_start_id=248088, slice_end_id=248089,
                    newline_id=198):
        merge=self.merge_factor
        token_counts=[int(h*w//(merge*merge)) for h,w in target_sizes]
        out=[image_start_id]+[image_token_id]*token_counts[0]+[image_end_id]
        rows,cols=map(int,grid)
        if self.slice_mode and rows and cols:
            for y in range(rows):
                if y: out.append(newline_id)
                for x in range(cols):
                    i=1+y*cols+x
                    out.extend([slice_start_id]+[image_token_id]*token_counts[i]+[slice_end_id])
        return out

    def placeholder_text(self, target_sizes, grid, image_token="<image>", image_start="<image>", image_end="</image>", slice_start="<slice>", slice_end="</slice>"):
        counts=[int(int(h)*int(w)//(self.merge_factor*self.merge_factor)) for h,w in target_sizes]
        text=image_start+image_token*counts[0]+image_end
        rows,cols=map(int,grid)
        if self.slice_mode and rows and cols:
            text += "\n".join((slice_start+image_token*counts[1]+slice_end)*cols for _ in range(rows))
        return text

    def canvas_position_ids(self, input_ids, mm_token_type_ids, target_sizes, grids,
                            image_token_id=248056, image_start_id=248078, image_end_id=248079,
                            slice_start_id=248088, slice_end_id=248089, newline_id=198):
        """Canvas M-RoPE positions for image placeholders, ported from HF 5.18."""
        ids=torch.as_tensor(input_ids,dtype=torch.long).flatten()
        types=torch.as_tensor(mm_token_type_ids,dtype=torch.long).flatten()
        if ids.numel()!=types.numel(): raise ValueError('input_ids/mm_token_type_ids length mismatch')
        pos=torch.arange(ids.numel(),dtype=torch.long).repeat(3,1)
        merge=self.merge_factor; crop_index=0; search=0
        while search < ids.numel():
            found=torch.nonzero(ids[search:]==image_start_id).flatten()
            if not len(found): break
            start=int(found[0])+search
            ends=torch.nonzero(ids[start+1:]==image_end_id).flatten()
            if not len(ends): raise ValueError('unclosed image_start marker')
            thumb_end=start+1+int(ends[0]); thumb_start=start+1
            spans=[]; i=thumb_end+1
            while i<ids.numel() and int(ids[i])==slice_start_id:
                a=i+1; b=a
                while b<ids.numel() and int(ids[b])==image_token_id: b+=1
                if b>=ids.numel() or int(ids[b])!=slice_end_id: raise ValueError('malformed slice placeholder')
                spans.append((a,b)); i=b+1
                if i<ids.numel() and int(ids[i])==newline_id: i+=1
                elif i<ids.numel() and int(ids[i])==slice_start_id: pass
                else: break
            frame_end=i
            ts=[tuple(map(int,x)) for x in target_sizes[crop_index:crop_index+1+len(spans)]]
            if len(ts)!=1+len(spans): raise ValueError('target_sizes crop count mismatch')
            image_number=int((ids[:start]==image_start_id).sum())
            rows,cols=map(int,grids[image_number])
            origin=start
            sh=ts[1][0]//merge if spans else 0; sw=ts[1][1]//merge if spans else 0
            ch=rows*sh if spans else ts[0][0]//merge
            cw=cols*sw if spans else ts[0][1]//merge
            pos[:,start:frame_end]=origin
            pos[1:,start]=max(origin-1,0)
            pos[1,thumb_end]=origin+ch; pos[2,thumb_end]=origin+cw
            def crop_positions(a,b,hw,hoff=0,woff=0,canvas_h=None,canvas_w=None):
                gh,gw=hw[0]//merge,hw[1]//merge
                hh=torch.linspace(0,(canvas_h if canvas_h is not None else gh)-1,gh).round().long()+hoff+origin
                ww=torch.linspace(0,(canvas_w if canvas_w is not None else gw)-1,gw).round().long()+woff+origin
                yy,xx=torch.meshgrid(hh,ww,indexing='ij')
                pos[:,a:b]=torch.stack((torch.full_like(yy,origin),yy,xx)).reshape(3,-1)
            crop_positions(thumb_start,thumb_end,ts[0],canvas_h=ch,canvas_w=cw)
            for k,(a,b) in enumerate(spans):
                oh=(k//cols)*sh; ow=(k%cols)*sw
                pos[1,a-1]=origin+oh; pos[2,a-1]=origin+ow
                pos[1,b]=origin+oh+sh-1; pos[2,b]=origin+ow+sw-1
                crop_positions(a,b,ts[k+1],oh,ow)
            for k in range(len(spans)-1):
                gap_start=spans[k][1]+1; gap_end=spans[k+1][0]-1
                if gap_end>gap_start:
                    pos[1,gap_start:gap_end]=origin+((k//cols)+1)*sh-1
                    pos[2,gap_start:gap_end]=origin+cols*sw
            next_pos=origin+max(ch,cw)+1
            if frame_end<ids.numel(): pos[:,frame_end:]=torch.arange(frame_end,ids.numel()).repeat(3,1)+(next_pos-frame_end)
            crop_index += 1+len(spans); search=frame_end
        return pos
