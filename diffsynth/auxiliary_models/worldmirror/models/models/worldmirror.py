from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .visual_transformer import VisualGeometryTransformer
from ..heads.camera_head import CameraHead
from ..heads.dense_head import DPTHead
from .rasterization import GaussianSplatRenderer
from ..utils.camera_utils import vector_to_camera_matrices, extrinsics_to_vector
from ..utils.priors import normalize_depth, normalize_poses

from huggingface_hub import PyTorchModelHubMixin
from diffsynth.models.utils import hash_state_dict_keys


class WorldMirror(nn.Module, PyTorchModelHubMixin):
    def __init__(self,
                 img_size=518,
                 patch_size=14,
                 embed_dim=1024,
                 gs_dim=256,
                 enable_cond=True,
                 enable_cam=True,
                 enable_pts=True,
                 enable_depth=True,
                 enable_norm=True,
                 enable_motion=True,
                 enable_gs=True,
                 enable_dynamic_gs_attr=True,
                 enable_waypoints=False,
                 n_waypoints=2,
                 waypoint_positions=(1.0 / 3.0, 2.0 / 3.0),
                 enable_motion_gate=False,
                 motion_gate_init_bias=4.0,
                 interpolation_mode=None,
                 life_span_gamma=10.0,
                 dynamic_threshold=0.0,
                 dynamic_threshold_time_mode="displacement",
                 enable_global_motion_tracking=False,
                 dynamic_threshold2=0.0,
                 occlusion_threshold=0.05,
                 bidirection=True,
                 patch_embed="dinov2_vitl14_reg",
                 fixed_patch_embed=False,
                 sampling_strategy="uniform",
                 dpt_gradient_checkpoint=False,
                 condition_strategy=["token", "pow3r", "token"],
                 **kwargs):

        super().__init__()
        # Configuration flags
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.gs_dim = gs_dim
        self.enable_cam = enable_cam
        self.enable_pts = enable_pts
        self.enable_depth = enable_depth
        self.enable_cond = enable_cond
        self.enable_norm = enable_norm
        self.enable_motion = enable_motion
        self.enable_gs = enable_gs
        self.enable_dynamic_gs_attr = enable_dynamic_gs_attr
        self.enable_waypoints = bool(enable_waypoints)
        self.n_waypoints = int(n_waypoints)
        self.waypoint_positions = tuple(float(p) for p in waypoint_positions)
        self.enable_motion_gate = bool(enable_motion_gate)
        self.motion_gate_init_bias = float(motion_gate_init_bias)
        if self.enable_waypoints:
            assert len(self.waypoint_positions) == self.n_waypoints, (
                f"len(waypoint_positions)={len(self.waypoint_positions)} "
                f"!= n_waypoints={self.n_waypoints}"
            )
            assert all(0.0 < p < 1.0 for p in self.waypoint_positions), (
                "waypoint_positions must lie strictly inside (0,1)"
            )
            assert all(
                a < b for a, b in zip(self.waypoint_positions[:-1], self.waypoint_positions[1:])
            ), "waypoint_positions must be strictly increasing"
        # Default interpolation_mode follows enable_waypoints if not given.
        if interpolation_mode is None:
            interpolation_mode = "cubic_waypoint" if self.enable_waypoints else "linear"
        assert interpolation_mode in ("linear", "cubic_waypoint")
        self.interpolation_mode = interpolation_mode
        self.life_span_gamma = life_span_gamma
        self.dynamic_threshold = dynamic_threshold
        self.dynamic_threshold_time_mode = dynamic_threshold_time_mode
        self.enable_global_motion_tracking = enable_global_motion_tracking
        self.dynamic_threshold2 = dynamic_threshold2
        self.occlusion_threshold = occlusion_threshold
        self.bidirection = bool(bidirection)
        self.patch_embed = patch_embed
        self.sampling = sampling_strategy
        self.dpt_checkpoint = dpt_gradient_checkpoint
        self.cond_methods = condition_strategy
        self.config = self._store_config()

        # Visual geometry transformer
        self.visual_geometry_transformer = VisualGeometryTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            enable_cond=enable_cond,
            enable_motion=enable_motion,
            sampling_strategy=sampling_strategy,
            patch_embed=patch_embed,
            fixed_patch_embed=fixed_patch_embed,
            condition_strategy=condition_strategy
        )

        # Initialize prediction heads
        self._init_heads(embed_dim, patch_size, gs_dim)

    def _store_config(self):
        """Save the model configuration"""
        return {
            "img_size": self.img_size,
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "gs_dim": self.gs_dim,
            "enable_cam": self.enable_cam,
            "enable_pts": self.enable_pts,
            "enable_depth": self.enable_depth,
            "enable_norm": self.enable_norm,
            "enable_gs": self.enable_gs,
            "patch_embed": self.patch_embed,
            "sampling_strategy": self.sampling,
            "dpt_checkpoint": self.dpt_checkpoint,
            "condition_strategy": self.cond_methods,
            # Codex B13: persist waypoint / interpolation options so hub or
            # checkpoint restores reconstruct the cubic-spline rasterizer.
            "enable_waypoints": self.enable_waypoints,
            "n_waypoints": self.n_waypoints,
            "waypoint_positions": list(self.waypoint_positions),
            "interpolation_mode": self.interpolation_mode,
            "bidirection": self.bidirection,
        }

    def _init_heads(self, dim, patch_size, gs_dim):
        """Initialize all prediction heads"""

        # Camera pose prediction head
        if self.enable_cam:
            self.cam_head = CameraHead(dim_in=2 * dim)

        # 3D point prediction head
        if self.enable_pts:
            self.pts_head = DPTHead(
                dim_in=2 * dim,
                output_dim=4,
                patch_size=patch_size,
                activation="inv_log+expp1"
            )

        # Depth prediction head
        if self.enable_depth:
            self.depth_head = DPTHead(
                dim_in=2 * dim,
                output_dim=2,
                patch_size=patch_size,
                activation="exp+expp1",
            )

        # Surface normal prediction head
        if self.enable_norm:
            self.norm_head = DPTHead(
                dim_in=2 * dim,
                output_dim=4,
                patch_size=patch_size,
                activation="norm+expp1",
            )

        # Velocity prediction heads
        if self.enable_motion:
            self.velocity_fwd_head = DPTHead(
                dim_in=dim,
                output_dim=4,
                patch_size=patch_size,
                activation="inv_log+expp1",
            )
            self.velocity_bwd_head = DPTHead(
                dim_in=dim,
                output_dim=4,
                patch_size=patch_size,
                activation="inv_log+expp1",
            )

        # Gaussian splatting feature head and renderer
        if self.enable_gs:
            self.gs_head = DPTHead(
                dim_in=2 * dim,
                output_dim=2,
                patch_size=patch_size,
                features=gs_dim,
                is_gsdpt=True,
                activation="exp+expp1"
            )
            self.gs_renderer = GaussianSplatRenderer(
                sh_degree=0,
                enable_prune=True,
                voxel_size=0.002,
                is_4dgs=self.enable_dynamic_gs_attr,
                life_span_gamma=self.life_span_gamma,
                dynamic_threshold=self.dynamic_threshold,
                dynamic_threshold_time_mode=self.dynamic_threshold_time_mode,
                global_motion_tracking=self.enable_global_motion_tracking,
                dynamic_threshold2=self.dynamic_threshold2,
                occlusion_threshold=self.occlusion_threshold,
                bidirection=self.bidirection,
                interpolation_mode=self.interpolation_mode,
                waypoint_positions=self.waypoint_positions,
            )
            # Dynamic Gaussian splatting attribute heads
            if self.enable_dynamic_gs_attr:
                self.gs_fwd_attr_head = DPTHead(
                    dim_in=dim,
                    output_dim=3,
                    patch_size=patch_size,
                    activation="rotation+none", # use 'none' to disable confidence prediction
                )
                self.gs_bwd_attr_head = DPTHead(
                    dim_in=dim,
                    output_dim=3,
                    patch_size=patch_size,
                    activation="rotation+none", # use 'none' to disable confidence prediction
                )

        # Multi-waypoint heads (Item B): predict per-Gaussian residual
        # displacement at intermediate u positions in (0,1) between consecutive
        # keyframes. Downstream adds the linear-motion baseline u * endpoint,
        # so zero residuals reproduce the pretrained linear reconstructor.
        # Output channels are 3*n_waypoints; downstream code reshapes to
        # [..., n_waypoints, 3]. Activation "linear+none" → no per-waypoint
        # confidence.
        if self.enable_waypoints:
            self.waypoint_fwd_head = DPTHead(
                dim_in=dim,
                output_dim=3 * self.n_waypoints,
                patch_size=patch_size,
                activation="linear+none",
            )
            self.waypoint_bwd_head = DPTHead(
                dim_in=dim,
                output_dim=3 * self.n_waypoints,
                patch_size=patch_size,
                activation="linear+none",
            )
            self._zero_waypoint_residual_head(self.waypoint_fwd_head)
            self._zero_waypoint_residual_head(self.waypoint_bwd_head)

        # Per-pixel motion gate: a fresh, full-rank DPTHead emits a 1-channel
        # logit per pixel; sigmoid -> a soft "is this a mover?" probability in
        # [0,1] that multiplies velocity AND waypoint displacement downstream, so
        # a closed gate (g->0) freezes that Gaussian without the global
        # dynamic_threshold band-aid. Init OPEN (bias -> g~1) so a fresh model
        # reproduces the ungated motion at step 0; the BCE loss vs the GT dynamic
        # mask then closes it on static pixels. (output_dim=1, "linear+none" ->
        # raw logit, no conf channel.)
        if self.enable_motion_gate:
            self.motion_gate_fwd_head = DPTHead(
                dim_in=dim, output_dim=1, patch_size=patch_size, activation="linear+none",
            )
            self.motion_gate_bwd_head = DPTHead(
                dim_in=dim, output_dim=1, patch_size=patch_size, activation="linear+none",
            )
            self._init_motion_gate_head_open(self.motion_gate_fwd_head, self.motion_gate_init_bias)
            self._init_motion_gate_head_open(self.motion_gate_bwd_head, self.motion_gate_init_bias)

        # P0 memory: thread DPT gradient-checkpointing to EVERY DPTHead (cuts the
        # dominant forward-held activation block; see DPTHead.scratch_forward).
        # Done post-construction so all heads — current and future — are covered
        # without editing each call site. No-op when dpt_gradient_checkpoint=False.
        for _m in self.modules():
            if isinstance(_m, DPTHead):
                _m.use_gradient_checkpoint = bool(self.dpt_checkpoint)

    @staticmethod
    def _init_motion_gate_head_open(head, bias=4.0):
        """Zero the gate head's final conv weight and set a positive bias so the
        gate starts OPEN (sigmoid(bias)~=1): a fresh model reproduces ungated
        motion at step 0, and BCE vs the GT dynamic mask then closes it on
        static pixels."""
        final = head.scratch.output_conv2[-1]
        if isinstance(final, nn.Conv2d):
            nn.init.zeros_(final.weight)
            if final.bias is not None:
                nn.init.constant_(final.bias, float(bias))

    @staticmethod
    def _zero_waypoint_residual_head(head):
        """Initialize waypoint residual heads to exact linear motion.

        P2 starts from a P1 checkpoint whose motion model is linear between
        adjacent context frames. The waypoint heads are new, so their random
        outputs must not perturb rendering at step 0. Zeroing the final
        prediction conv makes the residuals zero while keeping the head
        trainable.
        """
        final = head.scratch.output_conv2[-1]
        if isinstance(final, nn.Conv2d):
            nn.init.zeros_(final.weight)
            if final.bias is not None:
                nn.init.zeros_(final.bias)

    def forward(self, views: Dict[str, torch.Tensor], cond_flags: List[int]=[0, 0, 0], is_inference=True, use_motion=True, ledger_tokens=None, splats_correction=None):
        """
        Execute forward pass through the WorldMirror model.

        Args:
            views: Input data dictionary
            cond_flags: Conditioning flags [depth, rays, camera]
            ledger_tokens: Optional scene-ledger tokens (cross-window global
                module), threaded to the aggregator's gated ledger adapters.
            splats_correction: Optional fn(splats, predictions) applied to the
                per-pixel splat tensors before separate_splats builds the
                Gaussians lists (output-gauge correction; see rasterization).

        Returns:
            dict: Prediction results dictionary
        """
        imgs = views['img']

        # Enable conditional input during training if enabled, or during inference if any cond_flags are set
        use_cond = sum(cond_flags) > 0
        if (imgs.shape[1] == 1):
            use_motion = False

        # Extract priors and process features based on conditional input
        if use_cond:
            priors = self.extract_priors(views)
            token_list, patch_start_idx, fwd_token_list, bwd_token_list = self.visual_geometry_transformer(
                imgs, priors, cond_flags=cond_flags, use_motion=(use_motion and is_inference),
                ledger_tokens=ledger_tokens,
            )
        else:
            token_list, patch_start_idx, fwd_token_list, bwd_token_list = self.visual_geometry_transformer(
                imgs, use_motion=(use_motion and is_inference), ledger_tokens=ledger_tokens,
            )

        # Generate all predictions
        preds = self._gen_all_preds(
            token_list, imgs, patch_start_idx, views, cond_flags, is_inference, use_motion,
            fwd_token_list, bwd_token_list, ledger_tokens=ledger_tokens,
            splats_correction=splats_correction,
        )

        for key, value in preds.items():
            if isinstance(value, torch.Tensor) and value.dtype == torch.bfloat16:
                preds[key] = value.to(torch.float32)
            elif isinstance(value, list):
                for batch_value in value:
                    for frame_value in batch_value:
                        frame_value.to(torch.float32)
        return preds

    def _gen_all_preds(self, token_list, imgs, patch_start_idx,
                        views, cond_flags, is_inference, use_motion,
                       fwd_token_list=[], bwd_token_list=[], ledger_tokens=None,
                       splats_correction=None):
        """Generate all enabled predictions"""
        preds = {}

        # Pooled tap features for lightweight external heads (G2 gauge head):
        # mean over tokens per FRAME per tap layer -> [B, n_taps, S, 2C].
        # Cheap and read-only w.r.t. the token stream.
        preds["tap_pooled"] = torch.stack(
            [t.float().mean(dim=2) for t in token_list], dim=1
        )

        # Camera pose prediction
        if self.enable_cam:
            cam_seq = self.cam_head(token_list)
            cam_params = cam_seq[-1]
            preds["camera_params"] = cam_params
            c2w_mat, int_mat = self.transform_camera_vector(cam_params, imgs.shape[-2], imgs.shape[-1])
            preds["camera_poses"] = c2w_mat  # C2W pose (OpenCV) in world coordinates: [B, S, 4, 4]
            preds["camera_intrs"] = int_mat  # Camera intrinsic matrix: [B, S, 3, 3]

        # Depth prediction
        if self.enable_depth:
            depth, depth_conf = self.depth_head(
                token_list, images=imgs, patch_start_idx=patch_start_idx,
            )
            preds["depth"] = depth
            preds["depth_conf"] = depth_conf

        # 3D point prediction
        if self.enable_pts:
            pts, pts_conf = self.pts_head(
                token_list, images=imgs, patch_start_idx=patch_start_idx,
            )
            preds["pts3d"] = pts
            preds["pts3d_conf"] = pts_conf

        # Normal prediction
        if self.enable_norm:
            normals, norm_conf = self.norm_head(
                token_list, images=imgs, patch_start_idx=patch_start_idx,
            )
            preds["normals"] = normals
            preds["normals_conf"] = norm_conf

        # Prepare context predictions for motion and GS heads
        if self.enable_motion or self.enable_gs:
            context_preds = self.prepare_contexts(
                views, cond_flags, is_inference, use_motion, ledger_tokens=ledger_tokens
            )
        else:
            context_preds = {}

        fwd_token_list = context_preds.get("fwd_token_list", fwd_token_list)
        bwd_token_list = context_preds.get("bwd_token_list", bwd_token_list)

        # Velocity prediction
        if self.enable_motion and use_motion:
            assert len(fwd_token_list) > 0 and len(bwd_token_list) > 0
            vel_fwd, vel_fwd_conf = self.velocity_fwd_head(
                fwd_token_list,
                images=context_preds.get("imgs", imgs)[:, :-1],
                patch_start_idx=patch_start_idx
            )
            vel_bwd, vel_bwd_conf = self.velocity_bwd_head(
                bwd_token_list,
                images=context_preds.get("imgs", imgs)[:, 1:],
                patch_start_idx=patch_start_idx
            )
            preds["velocity_fwd"] = vel_fwd
            preds["velocity_fwd_conf"] = vel_fwd_conf
            preds["velocity_bwd"] = vel_bwd
            preds["velocity_bwd_conf"] = vel_bwd_conf

            # Per-pixel motion gate (raw logits; sigmoid + multiply happens in
            # prepare_splats). Same tokens / image slicing as the velocity heads.
            if self.enable_motion_gate:
                gate_fwd, _ = self.motion_gate_fwd_head(
                    fwd_token_list,
                    images=context_preds.get("imgs", imgs)[:, :-1],
                    patch_start_idx=patch_start_idx,
                )
                gate_bwd, _ = self.motion_gate_bwd_head(
                    bwd_token_list,
                    images=context_preds.get("imgs", imgs)[:, 1:],
                    patch_start_idx=patch_start_idx,
                )
                preds["motion_gate_fwd_logit"] = gate_fwd   # [B, S-1, H, W, 1] raw logit
                preds["motion_gate_bwd_logit"] = gate_bwd

        # 3D Gaussian Splatting
        if self.enable_gs:
            gs_feat, gs_depth, gs_depth_conf = self.gs_head(
                context_preds.get("token_list", token_list),
                images=context_preds.get("imgs", imgs),
                patch_start_idx=patch_start_idx
            )
            # Safety clamp on `gs_head`'s `exp()` output. Without this, LoRA
            # finetuning can push the exp pre-activation arbitrarily high,
            # producing `gs_depth >> 100` on unsupervised pixels (sky / GT
            # holes) that the downstream depth loss can't anchor. With
            # vggt-canonicalized data the legitimate range is ~[0.05, 15],
            # so 100 leaves ample headroom while bounding runaway drift.
            # See drender/m2_reconstructor/training/losses.py hard_max=100.
            gs_depth = gs_depth.clamp(min=1e-3, max=100.0)
            preds["gs_depth"] = gs_depth
            preds["gs_depth_conf"] = gs_depth_conf

            # Dynamic GS attributes
            if self.enable_dynamic_gs_attr and use_motion:
                assert len(fwd_token_list) > 0 and len(bwd_token_list) > 0
                gs_fwd_attr, _ = self.gs_fwd_attr_head(
                    fwd_token_list,
                    images=context_preds.get("imgs", imgs)[:, :-1],
                    patch_start_idx=patch_start_idx
                )
                gs_bwd_attr, _ = self.gs_bwd_attr_head(
                    bwd_token_list,
                    images=context_preds.get("imgs", imgs)[:, 1:],
                    patch_start_idx=patch_start_idx
                )
                preds["gs_fwd_attr"] = gs_fwd_attr
                preds["gs_bwd_attr"] = gs_bwd_attr

            # Multi-waypoint prediction (Item B): per-Gaussian displacement at
            # `n_waypoints` fixed u ∈ (0,1) positions between consecutive
            # keyframes, used by the rasterizer's natural-cubic spline (Item C).
            if self.enable_waypoints and use_motion:
                assert len(fwd_token_list) > 0 and len(bwd_token_list) > 0
                wp_fwd, _ = self.waypoint_fwd_head(
                    fwd_token_list,
                    images=context_preds.get("imgs", imgs)[:, :-1],
                    patch_start_idx=patch_start_idx,
                )
                wp_bwd, _ = self.waypoint_bwd_head(
                    bwd_token_list,
                    images=context_preds.get("imgs", imgs)[:, 1:],
                    patch_start_idx=patch_start_idx,
                )
                # Interior-knot RESIDUAL parameterization. The DPTHead emits
                # [..., 3*n_waypoints] (n_waypoints==2); the two slots are the
                # path's displacement residuals r1, r2 FROM THE STRAIGHT LINE
                # u·velocity at the fixed knots u=1/3 and u=2/3. The rasterizer
                # evaluates the unique cubic through (0,0), (1/3, e/3+r1),
                # (2/3, 2e/3+r2), (1, e):
                #     d(u) = u·e + b1(u)·r1 + b2(u)·r2
                # (b1/b2 in rasterization.py::_eval_cubic_segment). A zero
                # residual (the zero-init head) is exactly the straight line, so
                # a fresh P2 model matches P1 bit-for-bit at init; the dense
                # interior track loss drives r1/r2 directly — the supervised
                # quantity IS the head output, with no derived-tangent
                # indirection and no free end-tangent rotation to run away.
                fwd_shape = wp_fwd.shape[:-1] + (self.n_waypoints, 3)
                bwd_shape = wp_bwd.shape[:-1] + (self.n_waypoints, 3)
                preds["waypoints_fwd"] = wp_fwd.reshape(fwd_shape)
                preds["waypoints_bwd"] = wp_bwd.reshape(bwd_shape)

            preds = self.gs_renderer.render(
                gs_feats=gs_feat,
                images=imgs,
                predictions=preds,
                views=views,
                context_predictions=context_preds,
                is_inference=is_inference,
                splats_correction=splats_correction,
            )
        return preds

    def extract_priors(self, views):
        """
        Extract and normalize geometric priors.

        Args:
            views: Input view data dictionary.

        Returns:
            tuple: (depths, rays, poses) Normalized priors.
        """
        h, w = views['img'].shape[-2:]

        # Initialize prior variables
        depths = rays = poses = None

        # Extract camera pose
        if 'camera_poses' in views:
            extrinsics = views['camera_poses'][:, :, :3]
            extrinsics = normalize_poses(extrinsics)
            cam_params = extrinsics_to_vector(extrinsics)
            poses = cam_params[:, :, :7]  # Shape: [B, S, 7]

        # Extract depth map
        if 'depthmap' in views:
            depth_h, depth_w = views['depthmap'].shape[-2:]
            depths = views['depthmap']
            if depth_h != h or depth_w != w:  # Check if depth dimensions match target resolution
                try:
                    depths = F.interpolate(depths, size=(h, w), mode='bilinear', align_corners=False)
                except:
                    import pdb; pdb.set_trace()
            depths = normalize_depth(depths)  # Shape: [B, S, H, W]

        # Extract ray directions
        if 'camera_intrs' in views:
            intrinsics = views['camera_intrs'][:, :, :3, :3]
            fx, fy = intrinsics[:, :, 0, 0] / w, intrinsics[:, :, 1, 1] / h
            cx, cy = intrinsics[:, :, 0, 2] / w, intrinsics[:, :, 1, 2] / h
            rays = torch.stack([fx, fy, cx, cy], dim=-1)  # Shape: [B, S, 4]

        return (depths, rays, poses)

    def transform_camera_vector(self, camera_params, h, w):
        ext_mat, int_mat = vector_to_camera_matrices(
            camera_params, image_hw=(h, w)
        )
        # Create homogeneous transformation matrix
        homo_row = torch.tensor([0, 0, 0, 1], device=ext_mat.device).view(1, 1, 1, 4)
        homo_row = homo_row.repeat(ext_mat.shape[0], ext_mat.shape[1], 1, 1)
        w2c_mat = torch.cat([ext_mat, homo_row], dim=2)
        c2w_mat = torch.linalg.inv(w2c_mat)
        return c2w_mat, int_mat

    def prepare_contexts(self, views, cond_flags, is_inference, use_motion, ledger_tokens=None):
        # Generate context views predictions
        context_preds = {}
        # only for training or evaluation
        if is_inference:
            return context_preds

        assert self.enable_cam and (self.enable_motion or self.enable_gs)
        if 'is_target' not in views:
            context_nums = views['img'].shape[1]
        else:
            context_nums = (views['is_target'][0] == False).sum().item()
        context_imgs = views['img'][:, :context_nums]

        use_cond = sum(cond_flags) > 0

        # Extract context priors and process features based on context views
        if use_cond:
            priors = self.extract_priors(views)
            context_priors = (prior[:, :context_nums] if prior is not None else None for prior in priors)
            context_token_list, _, context_fwd_token_list, context_bwd_token_list = self.visual_geometry_transformer(
                context_imgs, context_priors, cond_flags=cond_flags, use_motion=use_motion,
                ledger_tokens=ledger_tokens,
            )
        else:
            context_token_list, _, context_fwd_token_list, context_bwd_token_list = self.visual_geometry_transformer(
                context_imgs, use_motion=use_motion, ledger_tokens=ledger_tokens,
            )

        # Execute predictions
        # Context camera pose prediction
        context_cam_seq = self.cam_head(context_token_list)
        context_cam_params = context_cam_seq[-1]
        context_c2w_mat, context_int_mat = self.transform_camera_vector(context_cam_params, context_imgs.shape[-2], context_imgs.shape[-1])
        context_preds['camera_poses'] = context_c2w_mat  # C2W pose (OpenCV) in world coordinates: [B, S, 4, 4]
        context_preds['camera_intrs'] = context_int_mat  # Camera intrinsic matrix: [B, S, 3, 3]
        context_preds['token_list'] = context_token_list
        context_preds['imgs'] = context_imgs
        context_preds['fwd_token_list'] = context_fwd_token_list
        context_preds['bwd_token_list'] = context_bwd_token_list

        return context_preds

    @staticmethod
    def state_dict_converter():
        return ModelDictConverter()


class ModelDictConverter:
    def __init__(self):
        pass

    def from_civitai(self, state_dict):
        if hash_state_dict_keys(state_dict) == '1a1d001a35f78f3a7796a1e719ead340':
            config = {
                "enable_norm": False,
                "strict_load": True,
                # "upcast_to_float32": True,
            }
        else:
            config = {}
        return state_dict, config
