#!/usr/bin/env python
"""
render.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import cv2
import yaml
import mediapy as media
import numpy as np
import open3d as o3d
import torch
import tyro
from rich.console import Console
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
from typing_extensions import Literal, assert_never

from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from chamfer_distance import ChamferDistance

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.configs.base_config import Config  # pylint: disable=unused-import
from nerfstudio.data.datamanagers.base_datamanager import AnnotatedDataParserUnion
from nerfstudio.data.dataparsers.sdfstudio_dataparser import SDFStudioDataParserConfig
from nerfstudio.utils import install_checks

CONSOLE = Console(width=120)

def _render_image(
    meshfile: Path,
    cameras: Cameras,
    output_dir: Path,
    rendered_output_names: str,
    rendered_resolution_scaling_factor: float = 1.0
) -> None:
    """Helper function to create images for evaluation"""
    CONSOLE.print("[bold green]Creating images")
    cameras.rescale_output_resolution(rendered_resolution_scaling_factor)

    width = cameras[0].width[0].item()
    height = cameras[0].height[0].item()

    ply = o3d.io.read_triangle_mesh(str(meshfile))
    ply.compute_vertex_normals()
    ply.paint_uniform_color([1, 1, 1])

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("rendering", width=width, height=height)

    vis.add_geometry(ply)
    vis.get_render_option().load_from_json("scripts/render.json")

    output_image_dir = output_dir #.parent / output_dir.stem
    for render_name in rendered_output_names:
        output_image_dir_cur = output_image_dir / render_name
        output_image_dir_cur.mkdir(parents=True, exist_ok=True)

    num_frames = cameras.size
    index = -1
    rendered_images = []

    def move_forward(vis):
        ctr = vis.get_view_control()
        nonlocal index
        nonlocal cameras
        nonlocal rendered_images
        if index >= 0:
            # images = []
            for render_name in rendered_output_names:
                output_image_dir_cur = output_image_dir / render_name

                if render_name == "normal":
                    vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Normal
                elif render_name == "rgb":
                    vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Color

                vis.capture_screen_image(str(output_image_dir_cur / f"{index:05d}.png"), True)

                # images.append(cv2.imread(str(output_image_dir_cur / f"{index:05d}.png"))[:, :, ::-1])
            # if merge_type == "concat":
            #     images = np.concatenate(images, axis=1)
            # elif merge_type == "half":
            #     mask = np.zeros_like(images[0])
            #     mask[:, : mask.shape[1] // 2, :] = 1
            #     images = images[0] * mask + images[1] * (1 - mask)
            # rendered_images.append(images)
        index = index + 1
        if index < num_frames:

            param = ctr.convert_to_pinhole_camera_parameters()
            camera = cameras[index]
            width = camera.width[0].item()
            height = camera.height[0].item()
            fx = camera.fx[0].item()
            fy = camera.fy[0].item()
            cx = camera.cx[0].item()
            cy = camera.cy[0].item()
            camera = cameras[index]

            param.intrinsic.set_intrinsics(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)

            extrinsic = np.eye(4)
            extrinsic[:3, :] = camera.camera_to_worlds.cpu().numpy()
            # extrinsic[:3, 1:3] *= -1
            param.extrinsic = np.linalg.inv(extrinsic)

            ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
        else:
            vis.register_animation_callback(None)
            vis.destroy_window()

        return False
    
    vis.register_animation_callback(move_forward)
    vis.run()

def _parse_camera(camera_meta, data_dir):
    indices = list(range(len(camera_meta["frames"])))

    image_filenames = []
    fx = []
    fy = []
    cx = []
    cy = []
    camera_to_worlds = []

    for i, frame in enumerate(camera_meta["frames"]):
        image_filename = data_dir / frame["rgb_path"]

        intrinsics = torch.tensor(frame["intrinsics"])
        camtoworld = torch.tensor(frame["camtoworld"])

        # append data
        image_filenames.append(image_filename)
        fx.append(intrinsics[0, 0])
        fy.append(intrinsics[1, 1])
        cx.append(intrinsics[0, 2])
        cy.append(intrinsics[1, 2])
        camera_to_worlds.append(camtoworld)
    
    fx = torch.stack(fx)
    fy = torch.stack(fy)
    cx = torch.stack(cx)
    cy = torch.stack(cy)
    camera_to_worlds = torch.stack(camera_to_worlds)

    # Convert from COLMAP's/OPENCV's camera coordinate system to nerfstudio
    camera_to_worlds[:, 0:3, 1:3] *= -1

    height, width = camera_meta["height"], camera_meta["width"]
    cameras = Cameras(
        fx=fx[indices],
        fy=fy[indices],
        cx=cx[indices],
        cy=cy[indices],
        height=height,
        width=width,
        camera_to_worlds=camera_to_worlds[indices, :3, :4],
        camera_type=CameraType.PERSPECTIVE,
    )

    return cameras


@dataclass
class RenderMeshEval:
    # Path to config YAML file.
    meshfile: Path
    # Config file
    load_config: Path
    # Ground Truth Point Cloud file
    pcfile: Path
    # Original dataset direction
    data_dir: Path
    # Filename of the camera path to render.
    camera_path_filename: Path
    # Name of the output file.
    output_dir: Path
    # Evaluation output format output file
    eval_output_path: Path
    # Name of the renderer outputs to use. rgb, depth, etc. concatenates them along y axis
    rendered_output_names: List[str] = field(default_factory=lambda: ["rgb", "normal"])
    #  Trajectory to render.
    traj: Literal["spiral", "filename", "interpolate", "ellipse"] = "filename"
    # Scaling factor to apply to the camera image resolution.
    downscale_factor: int = 1
    

    def main(self) -> None:
        """Main function."""

        install_checks.check_ffmpeg_installed()
        config = yaml.load(self.load_config.read_text(), Loader=yaml.Loader)

        if self.traj == "filename":
            self.camera_path_filename = self.data_dir / self.camera_path_filename
            with open(self.camera_path_filename, "r", encoding="utf-8") as f:
                meta = json.load(f)
                camera = _parse_camera(meta, self.data_dir) 
        else:
            raise NotImplementedError
        
        _render_image(
            meshfile=self.meshfile,
            cameras=camera,
            output_dir=self.output_dir,
            rendered_resolution_scaling_factor=1.0/self.downscale_factor,
            rendered_output_names=self.rendered_output_names
        )

        # evaluation
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()
        self.chamf = ChamferDistance()

        frames = meta["frames"]
        
        psnr_score, ssim_score, lpips_score = 0., 0., 0.

        eval_dict = {}
        frames_score = []
        for i in range(len(frames)):
            score_dict = {}

            gt_filename = self.data_dir / frames[i]["rgb_path"]
            output_filename = self.output_dir / "rgb" / f"{i:05d}.png"

            gt_img = cv2.imread(str(gt_filename))[..., :3] / 255.
            out_img = cv2.imread(str(output_filename))[..., :3] / 255.

            gt = torch.tensor(gt_img, dtype=torch.float32)
            out = torch.tensor(out_img, dtype=torch.float32)
            gt = torch.moveaxis(gt, -1, 0)[None, ...]
            out = torch.moveaxis(out, -1, 0)[None, ...]

            psnr = self.psnr(gt, out)
            ssim = self.ssim(gt, out)
            lpips = self.lpips(gt, out)

            psnr_score += psnr.item()
            ssim_score += ssim.item()
            lpips_score += lpips.item()

            score_dict["rgb_path"] = frames[i]["rgb_path"]
            score_dict["metrics"] = {}
            score_dict["metrics"]["psnr"] = float(psnr.item())
            score_dict["metrics"]["ssim"] = float(ssim)
            score_dict["metrics"]["lpips"] = float(lpips)

            frames_score.append(score_dict)
        
        eval_dict["frames"] = frames_score

        psnr_avg = psnr_score / len(frames)
        ssim_avg = ssim_score / len(frames)
        lpips_avg = lpips_score / len(frames)

        metrics = {
            "avg_psnr" : psnr_avg,
            "avg_ssim" : ssim_avg,
            "avg_lpips" : lpips_avg
        }
    
        eval_dict["avg_metrics"] = metrics

        # evaluate point cloud
        # ply = o3d.io.read_triangle_mesh(str(self.meshfile))
        # out_pc = ply.vertices

        # obj = o3d.io.read_point_cloud(str(self.pcfile))
        # gt_pc = obj.points

        # chamfer = self.chamf(gt_pc, out_pc)
        # eval_dict["metrics"]["chamfer_dist"] = chamfer

        with open(self.output_dir / self.eval_output_path, "w", encoding="utf-8") as f:
            json.dump(eval_dict, f, indent=2)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(RenderMeshEval).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(RenderMeshEval)  # noqa