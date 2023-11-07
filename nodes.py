from .consistencydecoder import ConsistencyDecoder
from PIL import Image
import torch
import numpy as np
import os
import folder_paths
import comfy
from diffusers import StableDiffusionPipeline

def conv_pil_tensor(img):
	return (torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0),)

pwd = os.getcwd()

class Consistency_Decoder:
	@classmethod
	def INPUT_TYPES(s):
		return {"required":{"latent": ("LATENT", ), "Consistency":("Consistency", )}}

	RETURN_TYPES = ("IMAGE",)
	FUNCTION = "decode"

	CATEGORY = "latent"

	def decode(self, latent,Consistency):
		#print(latent)
		# decoder_consistency = ConsistencyDecoder(device="cuda:0", download_root=pwd)
		consistent_latent = Consistency(latent["samples"].to("cuda:0"))
		del Consistency
		image = consistent_latent[0].cpu().numpy()
		image = (image + 1.0) * 127.5
		image = image.clip(0, 255).astype(np.uint8)
		image = Image.fromarray(image.transpose(1, 2, 0))
		return conv_pil_tensor(image)
	


class Consistency_load:
	@classmethod
	def INPUT_TYPES(s):
		return {"required": { "vae_name": (folder_paths.get_filename_list("vae"), )}}
	RETURN_TYPES = ("Consistency",)
	FUNCTION = "Consistency_load"

	CATEGORY = "loaders"

	def Consistency_load(self, vae_name):
		vae_path = folder_paths.get_full_path("vae", vae_name)
		# sd = comfy.utils.load_torch_file(vae_path)
		# vae = comfy.sd.VAE(sd=sd)
		# pipe.vae.cuda()
		# print(vae_path)
		Consistency = ConsistencyDecoder(device="cuda:0",path=vae_path)
		return (Consistency,)
	

# class Consistency_Encoder:
# 	@classmethod
# 	def INPUT_TYPES(s):
# 		return {"required": { "pixels": ("IMAGE", ), "decoder_consistency": ("decoder_consistency", )}}
# 	RETURN_TYPES = ("LATENT",)
# 	FUNCTION = "encode"

# 	CATEGORY = "latent"

# 	@staticmethod
# 	def 




NODE_CLASS_MAPPINGS = {
	"Consistency_decoer": Consistency_Decoder,
	"Consistency_load": Consistency_load,
}

NODE_DISPLAY_NAME_MAPPINGS = {
	"Comfy_ConsistencyVAE": "Consistency VAE Decoder",
	"Consistency_load": "Consistency Load",
}
