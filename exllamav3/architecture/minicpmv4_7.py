from __future__ import annotations
from types import SimpleNamespace

from .qwen3_5 import Qwen3_5BaseModel, Qwen3_5VLMoeBaseConfig
from .minicpmv4_7_vision import MiniCPMV4_7VisionModel
from ..util.rope import RoPE
import torch


class MiniCPMV4_7CanvasRoPE(RoPE):
    """Generator-side canvas M-RoPE computed from image MMEmbedding spans."""
    def get_mrope_freqs(self,input_ids,embeddings,max_length):
        from .mm_processing.minicpmv4_7 import MiniCPMV4_7ImageProcessor
        ids=input_ids.squeeze(0).contiguous().clone()
        types=torch.zeros_like(ids)
        sizes=[]; grids=[]
        for e in embeddings:
            ids[(ids>=e.first_index)&(ids<e.last_index)]=int(e.metadata['image_token_id'])
            types[(input_ids.squeeze(0)>=e.first_index)&(input_ids.squeeze(0)<e.last_index)]=1
            sizes.extend(e.metadata['target_sizes']); grids.extend(e.metadata['grids'])
        proc=MiniCPMV4_7ImageProcessor(downsample_mode=embeddings[0].metadata.get('downsample_mode','16x'))
        pos=proc.canvas_position_ids(ids,types,torch.tensor(sizes),torch.tensor(grids),
            image_token_id=int(embeddings[0].metadata['image_token_id']),
            image_start_id=int(embeddings[0].metadata['image_start_id']),
            image_end_id=int(embeddings[0].metadata['image_end_id']),
            slice_start_id=int(embeddings[0].metadata['slice_start_id']),
            slice_end_id=int(embeddings[0].metadata['slice_end_id']),
            newline_id=int(embeddings[0].metadata['newline_id']))
        next_pos=int(pos.max())+1
        mpos=torch.zeros((3,max_length),dtype=torch.long)
        mpos[:,:ids.numel()]=pos
        if max_length>ids.numel():
            cont=torch.arange(max_length-ids.numel(),dtype=torch.long)+next_pos
            mpos[:,ids.numel():]=cont
        inv=self.inv_freq[None,None,:,None].float().expand(3,1,-1,1)
        pp=mpos[:,None,None,:].float()
        freqs=(inv@pp).transpose(2,3)
        ft=freqs[0]
        if self.mrope_interleaved:
            for dim,offset in enumerate((1,2),start=1):
                idx=slice(offset,self.mrope_section[dim]*3,3); ft[...,idx]=freqs[dim,...,idx]
        else:
            start=self.mrope_section[0]
            for dim in (1,2):
                end=start+self.mrope_section[dim]; ft[...,start:end]=freqs[dim,...,start:end]; start=end
        return ft.contiguous(),next_pos


class MiniCPMV4_7TextModel(Qwen3_5BaseModel):
    """Qwen3.5-MoE language stack used by MiniCPM-V 4.7."""
    def __init__(self, config: MiniCPMV4_7Config, **kwargs):
        super().__init__(config=config,key_prefix="model.language_model",use_moe=True,**kwargs)
        self.g_rope=MiniCPMV4_7CanvasRoPE("cpu",config.rope_settings)
        # vocab_size 248144 is not a multiple of 128: the head is zero-padded to 248192, trim the logits back
        # to the true vocabulary so nothing downstream sees pad columns (from r0b0tlab/exllamav3@4d083d25)
        self.modules[self.logit_layer_idx].trim_padded_out = True


class MiniCPMV4_7Config(Qwen3_5VLMoeBaseConfig):
    """MiniCPM-V 4.7 config with its native NaViT-SigLIP vision component."""
    arch_string = "MiniCPMV4_7ForConditionalGeneration"

    def __init__(self, directory: str, **kwargs):
        super().__init__(directory,text_cfg="text_config",text_model=MiniCPMV4_7TextModel,
                         vision_model=None,mtp_model=None,**kwargs)
        v=self.read_cfg(dict,"vision_config")
        self.vision=SimpleNamespace(**v)
        self.vision.deepstack_visual_indexes=[]
        self.vision.insert_layer_id=self.read_cfg(int,"insert_layer_id",6)
        self.vision.window_kernel_size=(2,2)
        self.vision.window_hidden_size=self.vision.hidden_size*4
        self.vision.window_intermediate_size=self.vision.intermediate_size*4
        self.model_classes["vision"]=MiniCPMV4_7VisionModel
        self.downsample_mode=self.read_cfg(str,"downsample_mode","16x")
        self.mrope_mode=self.read_cfg(str,"mrope_mode","canvas")
        self.image_token_id=self.read_cfg(int,"image_token_id")
        self.video_token_id=self.read_cfg(int,"video_token_id")
        self.image_start_id=self.read_cfg(int,"image_start_id")
        self.image_end_id=self.read_cfg(int,"image_end_id")
        self.slice_start_id=self.read_cfg(int,"slice_start_id")
        self.slice_end_id=self.read_cfg(int,"slice_end_id")
        self.newline_id=self.read_cfg(int,"newline_id",198)
        self.merger_times=1
        self.merge_kernel_size=(2,2)


MiniCPMV4_7TextModel.config_class=MiniCPMV4_7Config
