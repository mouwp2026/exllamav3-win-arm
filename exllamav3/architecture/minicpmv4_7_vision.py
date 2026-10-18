from __future__ import annotations

from typing_extensions import override
import torch
import torch.nn.functional as F
from types import SimpleNamespace

from ..model.model import Model
from ..modules import Conv, PosEmbedding, TransformerBlock, Attention, Linear, LayerNorm
from ..util.rope import RopeSettings


class MiniCPMV4_7VisionModel(Model):
    """NaViT SigLIP tower with layer-6 2x2 window merger and final MLP merger."""
    @staticmethod
    @override
    def get_additional_compiled_tensors(config):
        return config.stc.list_tensors(prefix="model.vision_tower") | config.stc.list_tensors(prefix="model.merger")

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.config=config
        v=config.vision
        self.caps.update({"image_input":True, "default_vision_bits":16})
        self.patch=Conv(config,key="model.vision_tower.embeddings.patch_embedding",in_channels=3,
                        out_channels=v.hidden_size,kernel_size=(v.patch_size,v.patch_size))
        self.pos=PosEmbedding(config,key="model.vision_tower.embeddings.position_embedding",hidden_size=v.hidden_size)
        self.blocks=[]
        for i in range(v.num_hidden_layers):
            prefix=f"model.vision_tower.encoder.layers.{i}"
            block=TransformerBlock(config,key=prefix,layer_idx=i,
                attn_norm=LayerNorm(config,key=prefix+".layer_norm1",layernorm_eps=v.layer_norm_eps),
                attn=Attention(config,key=prefix+".self_attn",layer_idx=i,hidden_size=v.hidden_size,
                    head_dim=v.hidden_size//v.num_attention_heads,num_q_heads=v.num_attention_heads,
                    num_kv_heads=v.num_attention_heads,rope_settings=None,key_q="q_proj",key_k="k_proj",
                    key_v="v_proj",key_o="out_proj",qmap="block.attn",use_cu_seqlens=False),
                mlp_norm=LayerNorm(config,key=prefix+".layer_norm2",layernorm_eps=v.layer_norm_eps),
                mlp=__import__("exllamav3.modules",fromlist=["MLP"]).MLP(config,key=prefix+".mlp",
                    hidden_size=v.hidden_size,intermediate_size=v.intermediate_size,key_up="fc1",key_down="fc2",
                    activation_fn="gelu",qmap="block.mlp"))
            self.blocks.append(block)
        self.vit_ln=LayerNorm(config,key="model.vision_tower.vit_merger.layer_norm1",layernorm_eps=v.layer_norm_eps)
        self.vit_attn=Attention(config,key="model.vision_tower.vit_merger.self_attn",layer_idx=27,
            hidden_size=v.hidden_size,head_dim=v.hidden_size//v.num_attention_heads,num_q_heads=v.num_attention_heads,
            num_kv_heads=v.num_attention_heads,rope_settings=None,key_q="q_proj",key_k="k_proj",key_v="v_proj",
            key_o="out_proj",qmap="block.attn")
        self.vit_pre=LayerNorm(config,key="model.vision_tower.vit_merger.pre_norm",layernorm_eps=v.layer_norm_eps)
        self.vit_fc1=Linear(config,key="model.vision_tower.vit_merger.linear_1",in_features=v.hidden_size*4,
                            out_features=v.intermediate_size*4,qmap="block.mlp")
        self.vit_fc2=Linear(config,key="model.vision_tower.vit_merger.linear_2",in_features=v.intermediate_size*4,
                            out_features=v.hidden_size,qmap="block.mlp")
        self.post=LayerNorm(config,key="model.vision_tower.post_layernorm",layernorm_eps=v.layer_norm_eps)
        self.merger_pre=LayerNorm(config,key="model.merger.mlp.0.pre_norm",layernorm_eps=1e-6)
        self.merger_fc1=Linear(config,key="model.merger.mlp.0.linear_1",in_features=v.hidden_size*4,
                               out_features=v.hidden_size*4,qmap="block.mlp")
        self.merger_fc2=Linear(config,key="model.merger.mlp.0.linear_2",in_features=v.hidden_size*4,
                               out_features=config.hidden_size,qmap="block.mlp")
        self.modules=[self.patch,self.pos,*self.blocks,self.vit_ln,self.vit_attn,self.vit_pre,self.vit_fc1,
                      self.vit_fc2,self.post,self.merger_pre,self.merger_fc1,self.merger_fc2]
        self.merge_factor=4

    @staticmethod
    def _unpatchify(patches, h, w, p):
        n=h*w
        cols=patches[0].reshape(3,p,n,p).permute(0,1,3,2).reshape(1,3*p*p,n)
        return F.fold(cols,output_size=(h*p,w*p),kernel_size=p,stride=p)

    def _add_positions(self, x, h, w):
        side=self.config.vision.image_size//self.config.vision.patch_size
        device=x.device
        limits=torch.arange(1/side,1.0,1/side,device=device)
        yy=torch.bucketize(torch.arange(h,device=device)/h,limits,right=True)
        xx=torch.bucketize(torch.arange(w,device=device)/w,limits,right=True)
        ids=(yy[:,None]*side+xx[None,:]).reshape(-1)
        return x+self.pos.embedding.weight[ids].unsqueeze(0).to(x.dtype)

    def _window_merge(self,x,h,w,params):
        # Order windows row-major, pixels inside each window row-major.
        c=x.shape[-1]
        residual=x.reshape(1,h//2,2,w//2,2,c).permute(0,1,3,2,4,5).reshape(1,-1,4,c)
        y=self.vit_ln.forward(residual.reshape(-1,4,c),params)
        y=self.vit_attn.forward(y,params)
        y=residual.reshape(-1,4,c)+y
        flat=y.reshape(-1,4*c)
        r=y.mean(dim=1)
        z=self.vit_pre.forward(flat,params)
        z=self.vit_fc1.forward(z,params)
        z=F.gelu(z,approximate="tanh")
        z=self.vit_fc2.forward(z,params)
        return (z+r).reshape(1,-1,c)

    def _final_merge(self,x,h,w,params):
        c=x.shape[-1]
        y=x.reshape(1,h//2,2,w//2,2,c).permute(0,1,3,2,4,5).reshape(-1,4*c)
        y=self.merger_pre.forward(y,params)
        y=self.merger_fc1.forward(y,params)
        y=F.gelu(y)
        y=self.merger_fc2.forward(y,params)
        return y

    @override
    def forward(self, pixel_values, target_sizes, params=None, out_dtype=torch.float16):
        params={"causal":False,**(params or {})}
        v=self.config.vision; p=v.patch_size; chunks=[]; begin=0
        for h,w in target_sizes.tolist():
            n=int(h*w); crop=pixel_values[...,begin:begin+n*p]; begin+=n*p
            image=self._unpatchify(crop,int(h),int(w),p).to(self.patch.device)
            x=self.patch.forward(image,params)
            x=self._add_positions(x,int(h),int(w))
            for i,block in enumerate(self.blocks):
                x=block.forward(x,params)
                if i==v.insert_layer_id:
                    x=self._window_merge(x,int(h),int(w),params); h//=2; w//=2
            x=self.post.forward(x,params)
            chunks.append(self._final_merge(x,int(h),int(w),params))
        return torch.cat(chunks,dim=0)

    def get_image_embeddings(self, tokenizer, image, text_alias=None):
        """Create MMEmbedding objects carrying the exact crop token layout and features."""
        from ..tokenizer import MMEmbedding
        from .mm_processing.minicpmv4_7 import MiniCPMV4_7ImageProcessor
        prep=MiniCPMV4_7ImageProcessor()
        if isinstance(image,list):
            if text_alias is not None: raise ValueError('text_alias cannot be shared across images')
            return [self.get_image_embeddings(tokenizer,im) for im in image]
        batch=prep(image)
        emb=self.forward(batch.pixel_values.to(self.patch.device),batch.target_sizes.to(self.patch.device),
                         params={"causal":False}).cpu()
        ph=prep.placeholder(batch.target_sizes.tolist(),batch.grids[0].tolist(),
            image_token_id=self.config.image_token_id,image_start_id=self.config.image_start_id,
            image_end_id=self.config.image_end_id,slice_start_id=self.config.slice_start_id,
            slice_end_id=self.config.slice_end_id,newline_id=self.config.newline_id)
        token_string=torch.tensor([[(-1 if tid==self.config.image_token_id else tid) for tid in ph]],dtype=torch.long)
        mme=MMEmbedding(embeddings=emb,token_string=token_string,text_alias=text_alias)
        mme.metadata.update({"original_size":image.size,"target_sizes":batch.target_sizes.tolist(),
                            "grids":batch.grids.tolist(),"model_architecture":self.config.architecture,
                            "image_token_id":self.config.image_token_id,"image_start_id":self.config.image_start_id,
                            "image_end_id":self.config.image_end_id,"slice_start_id":self.config.slice_start_id,
                            "slice_end_id":self.config.slice_end_id,"newline_id":self.config.newline_id,
                            "downsample_mode":self.config.downsample_mode})
        mme.grid_thw=tuple(batch.target_sizes[0].tolist())
        mme.mrope_merge_size=4
        return mme

    def prepare_image_prompt(self, tokenizer, image, prompt, text_alias="<|image_here|>"):
        """Return (input_ids, image MMEmbedding, canvas 3D position_ids) for one templated image prompt.

        The prompt must contain text_alias where the image appears. Special chat-template tokens are
        encoded by tokenizer.encode(..., encode_special_tokens=True).
        """
        from .mm_processing.minicpmv4_7 import MiniCPMV4_7ImageProcessor
        mme=self.get_image_embeddings(tokenizer,image,text_alias=text_alias)
        batch=MiniCPMV4_7ImageProcessor()(image)
        input_ids=tokenizer.encode(prompt,encode_special_tokens=True,embeddings=[mme])
        ids=input_ids.clone()
        mm_type=torch.zeros_like(ids)
        mm_mask=(ids>=mme.first_index)&(ids<mme.last_index)
        ids[mm_mask]=self.config.image_token_id
        mm_type[mm_mask]=1
        positions=MiniCPMV4_7ImageProcessor().canvas_position_ids(ids[0],mm_type[0],batch.target_sizes,batch.grids,
            image_token_id=self.config.image_token_id,image_start_id=self.config.image_start_id,
            image_end_id=self.config.image_end_id,slice_start_id=self.config.slice_start_id,
            slice_end_id=self.config.slice_end_id,newline_id=self.config.newline_id).unsqueeze(1)
        return input_ids,mme,positions
