# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations
from typing import Any
import torch
import numpy as np
from diffusers import StableDiffusion3ControlNetPipeline, SD3ControlNetModel, SD3MultiControlNetModel
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.controlnet_sd3.pipeline_stable_diffusion_3_controlnet import retrieve_timesteps

from PIL import Image
import copy

from typing import Union, List, Optional, Callable, Dict

from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import calculate_shift


class StableDiffusion3BalancedPipeline:
    def __init__(self, pipeline, handler) -> None:
        self.pipeline = pipeline
        self.handler = handler


    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        skip_guidance_layers: List[int] = None,
        skip_layer_guidance_scale: float = 2.8,
        skip_layer_guidance_stop: float = 0.2,
        skip_layer_guidance_start: float = 0.01,
        mu: Optional[float] = None,
    ):
            r"""
            Function invoked when calling the pipeline for generation.

            Args:
                prompt (`str` or `List[str]`, *optional*):
                    The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                    instead.
                prompt_2 (`str` or `List[str]`, *optional*):
                    The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                    will be used instead
                prompt_3 (`str` or `List[str]`, *optional*):
                    The prompt or prompts to be sent to `tokenizer_3` and `text_encoder_3`. If not defined, `prompt` is
                    will be used instead
                height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                    The height in pixels of the generated image. This is set to 1024 by default for the best results.
                width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                    The width in pixels of the generated image. This is set to 1024 by default for the best results.
                num_inference_steps (`int`, *optional*, defaults to 50):
                    The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                    expense of slower inference.
                sigmas (`List[float]`, *optional*):
                    Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                    their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                    will be used.
                guidance_scale (`float`, *optional*, defaults to 7.0):
                    Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                    `guidance_scale` is defined as `w` of equation 2. of [Imagen
                    Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                    1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                    usually at the expense of lower image quality.
                negative_prompt (`str` or `List[str]`, *optional*):
                    The prompt or prompts not to guide the image generation. If not defined, one has to pass
                    `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                    less than `1`).
                negative_prompt_2 (`str` or `List[str]`, *optional*):
                    The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                    `text_encoder_2`. If not defined, `negative_prompt` is used instead
                negative_prompt_3 (`str` or `List[str]`, *optional*):
                    The prompt or prompts not to guide the image generation to be sent to `tokenizer_3` and
                    `text_encoder_3`. If not defined, `negative_prompt` is used instead
                num_images_per_prompt (`int`, *optional*, defaults to 1):
                    The number of images to generate per prompt.
                generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                    One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                    to make generation deterministic.
                latents (`torch.FloatTensor`, *optional*):
                    Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                    generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                    tensor will ge generated by sampling using the supplied random `generator`.
                prompt_embeds (`torch.FloatTensor`, *optional*):
                    Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                    provided, text embeddings will be generated from `prompt` input argument.
                negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                    Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                    weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                    argument.
                pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                    Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                    If not provided, pooled text embeddings will be generated from `prompt` input argument.
                negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                    Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                    weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                    input argument.
                ip_adapter_image (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
                ip_adapter_image_embeds (`torch.Tensor`, *optional*):
                    Pre-generated image embeddings for IP-Adapter. Should be a tensor of shape `(batch_size, num_images,
                    emb_dim)`. It should contain the negative image embedding if `do_classifier_free_guidance` is set to
                    `True`. If not provided, embeddings are computed from the `ip_adapter_image` input argument.
                output_type (`str`, *optional*, defaults to `"pil"`):
                    The output format of the generate image. Choose between
                    [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
                return_dict (`bool`, *optional*, defaults to `True`):
                    Whether or not to return a [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] instead of
                    a plain tuple.
                joint_attention_kwargs (`dict`, *optional*):
                    A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                    `self.processor` in
                    [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
                callback_on_step_end (`Callable`, *optional*):
                    A function that calls at the end of each denoising steps during the inference. The function is called
                    with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                    callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                    `callback_on_step_end_tensor_inputs`.
                callback_on_step_end_tensor_inputs (`List`, *optional*):
                    The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                    will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                    `._callback_tensor_inputs` attribute of your pipeline class.
                max_sequence_length (`int` defaults to 256): Maximum sequence length to use with the `prompt`.
                skip_guidance_layers (`List[int]`, *optional*):
                    A list of integers that specify layers to skip during guidance. If not provided, all layers will be
                    used for guidance. If provided, the guidance will only be applied to the layers specified in the list.
                    Recommended value by StabiltyAI for Stable Diffusion 3.5 Medium is [7, 8, 9].
                skip_layer_guidance_scale (`int`, *optional*): The scale of the guidance for the layers specified in
                    `skip_guidance_layers`. The guidance will be applied to the layers specified in `skip_guidance_layers`
                    with a scale of `skip_layer_guidance_scale`. The guidance will be applied to the rest of the layers
                    with a scale of `1`.
                skip_layer_guidance_stop (`int`, *optional*): The step at which the guidance for the layers specified in
                    `skip_guidance_layers` will stop. The guidance will be applied to the layers specified in
                    `skip_guidance_layers` until the fraction specified in `skip_layer_guidance_stop`. Recommended value by
                    StabiltyAI for Stable Diffusion 3.5 Medium is 0.2.
                skip_layer_guidance_start (`int`, *optional*): The step at which the guidance for the layers specified in
                    `skip_guidance_layers` will start. The guidance will be applied to the layers specified in
                    `skip_guidance_layers` from the fraction specified in `skip_layer_guidance_start`. Recommended value by
                    StabiltyAI for Stable Diffusion 3.5 Medium is 0.01.
                mu (`float`, *optional*): `mu` value used for `dynamic_shifting`.

            Examples:

            Returns:
                [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] or `tuple`:
                [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] if `return_dict` is True, otherwise a
                `tuple`. When returning a tuple, the first element is a list with the generated images.
            """

            height = height or self.pipeline.default_sample_size * self.pipeline.vae_scale_factor
            width = width or self.pipeline.default_sample_size * self.pipeline.vae_scale_factor

            # 1. Check inputs. Raise error if not correct
            self.pipeline.check_inputs(
                prompt,
                prompt_2,
                prompt_3,
                height,
                width,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                negative_prompt_3=negative_prompt_3,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                max_sequence_length=max_sequence_length,
            )

            self.pipeline._guidance_scale = guidance_scale
            self.pipeline._skip_layer_guidance_scale = skip_layer_guidance_scale
            self.pipeline._clip_skip = clip_skip
            self.pipeline._joint_attention_kwargs = joint_attention_kwargs
            self.pipeline._interrupt = False

            # 2. Define call parameters
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            device = self.pipeline._execution_device

            lora_scale = (
                self.pipeline.joint_attention_kwargs.get("scale", None) if self.pipeline.joint_attention_kwargs is not None else None
            )
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipeline.encode_prompt(
                prompt=prompt,
                prompt_2=prompt_2,
                prompt_3=prompt_3,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                negative_prompt_3=negative_prompt_3,
                do_classifier_free_guidance=self.pipeline.do_classifier_free_guidance,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=device,
                clip_skip=self.pipeline.clip_skip,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )

            if self.pipeline.do_classifier_free_guidance:
                if skip_guidance_layers is not None:
                    original_prompt_embeds = prompt_embeds
                    original_pooled_prompt_embeds = pooled_prompt_embeds
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

            # 4. Prepare latent variables
            num_channels_latents = self.pipeline.transformer.config.in_channels
            latents = self.pipeline.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )

            # 5. Prepare timesteps
            scheduler_kwargs = {}
            if self.pipeline.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
                _, _, height, width = latents.shape
                image_seq_len = (height // self.pipeline.transformer.config.patch_size) * (
                    width // self.pipeline.transformer.config.patch_size
                )
                mu = calculate_shift(
                    image_seq_len,
                    self.pipeline.scheduler.config.base_image_seq_len,
                    self.pipeline.scheduler.config.max_image_seq_len,
                    self.pipeline.scheduler.config.base_shift,
                    self.pipeline.scheduler.config.max_shift,
                )
                scheduler_kwargs["mu"] = mu
            elif mu is not None:
                scheduler_kwargs["mu"] = mu
            timesteps, num_inference_steps = retrieve_timesteps(
                self.pipeline.scheduler,
                num_inference_steps,
                device,
                sigmas=sigmas,
                **scheduler_kwargs,
            )
            num_warmup_steps = max(len(timesteps) - num_inference_steps * self.pipeline.scheduler.order, 0)
            self.pipeline._num_timesteps = len(timesteps)

            # 6. Prepare image embeddings
            if (ip_adapter_image is not None and self.pipeline.is_ip_adapter_active) or ip_adapter_image_embeds is not None:
                ip_adapter_image_embeds = self.pipeline.prepare_ip_adapter_image_embeds(
                    ip_adapter_image,
                    ip_adapter_image_embeds,
                    device,
                    batch_size * num_images_per_prompt,
                    self.pipeline.do_classifier_free_guidance,
                )

                if self.pipeline.joint_attention_kwargs is None:
                    self.pipeline._joint_attention_kwargs = {"ip_adapter_image_embeds": ip_adapter_image_embeds}
                else:
                    self.pipeline._joint_attention_kwargs.update(ip_adapter_image_embeds=ip_adapter_image_embeds)

            # 7. Denoising loop
            with self.pipeline.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if self.pipeline.interrupt:
                        continue

                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if self.pipeline.do_classifier_free_guidance else latents
                    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                    timestep = t.expand(latent_model_input.shape[0])

                    self.handler.update_adain_layers(t)

                    noise_pred = self.pipeline.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        joint_attention_kwargs=self.pipeline.joint_attention_kwargs,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if self.pipeline.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.pipeline.guidance_scale * (noise_pred_text - noise_pred_uncond)
                        should_skip_layers = (
                            True
                            if i > num_inference_steps * skip_layer_guidance_start
                            and i < num_inference_steps * skip_layer_guidance_stop
                            else False
                        )
                        if skip_guidance_layers is not None and should_skip_layers:
                            timestep = t.expand(latents.shape[0])
                            latent_model_input = latents
                            noise_pred_skip_layers = self.pipeline.transformer(
                                hidden_states=latent_model_input,
                                timestep=timestep,
                                encoder_hidden_states=original_prompt_embeds,
                                pooled_projections=original_pooled_prompt_embeds,
                                joint_attention_kwargs=self.pipeline.joint_attention_kwargs,
                                return_dict=False,
                                skip_layers=skip_guidance_layers,
                            )[0]
                            noise_pred = (
                                noise_pred + (noise_pred_text - noise_pred_skip_layers) * self.pipeline._skip_layer_guidance_scale
                            )

                    # compute the previous noisy sample x_t -> x_t-1
                    latents_dtype = latents.dtype
                    latents = self.pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                    if latents.dtype != latents_dtype:
                        if torch.backends.mps.is_available():
                            # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                            latents = latents.to(latents_dtype)

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(pipeline, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                        negative_pooled_prompt_embeds = callback_outputs.pop(
                            "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                        )

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.pipeline.scheduler.order == 0):
                        progress_bar.update()

            if output_type == "latent":
                image = latents

            else:
                latents = (latents / self.pipeline.vae.config.scaling_factor) + self.pipeline.vae.config.shift_factor

                image = self.pipeline.vae.decode(latents, return_dict=False)[0]
                image = self.pipeline.image_processor.postprocess(image, output_type=output_type)

            # Offload all models
            self.pipeline.maybe_free_model_hooks()

            return image