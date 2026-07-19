from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from tuning.common import TuningConfigError, load_tuning_config
from tuning.dreamzero import build_dreamzero_command
from tuning.fastwam import build_fastwam_command
from tuning.lingbot_va import build_lingbot_va_command


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TuningTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "fastwam"
        (self.repo / "scripts").mkdir(parents=True)
        (self.repo / "src").mkdir()
        self.python = self.root / "python"
        self.python.write_text("fixture")
        self.data = self.root / "data"
        self.data.mkdir()
        self.manifest = self.data / "cases.jsonl"
        self.manifest.write_text("{}\n")
        self.stats = self.data / "stats.json"
        self.stats.write_text("{}\n")
        self.models = self.root / "models"
        self.models.mkdir()
        self.stage1 = self.root / "stage1.pt"
        self.stage1.write_text("fixture")
        self.action = self.root / "action.pt"
        self.action.write_text("fixture")
        self.rectification = self.root / "rectification.json"
        self.rectification.write_text("{}\n")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_config(self) -> Path:
        config = {
            "schema_version": "wm-tuning-v1",
            "models": {
                "fastwam": {
                    "repo": "${TEST_FASTWAM_REPO}",
                    "python": str(self.python),
                    "data_root": str(self.data),
                    "case_manifest": str(self.manifest),
                    "normalization_stats": str(self.stats),
                    "text_embedding_cache_dir": str(self.data / "text"),
                    "diffsynth_model_base": str(self.models),
                    "stage1_video_checkpoint": str(self.stage1),
                    "action_dit_checkpoint": str(self.action),
                    "phases": {
                        "stage3_finetune": {
                            "task": "stage3_robocoin_memory_smoke",
                            "allowed_modes": ["joint_video_action"],
                            "include_robot_supervision": True,
                            "skip_dit_load_from_pretrain": True,
                            "initial_checkpoint": str(self.root / "stage2.pt"),
                        },
                        "overfit": {
                            "task": "stage3_robocoin_memory_smoke",
                            "allowed_modes": ["joint_video_action"],
                            "include_robot_supervision": True,
                            "skip_dit_load_from_pretrain": True,
                            "initial_checkpoint": str(self.root / "stage2.pt"),
                            "sample_offset": 1,
                            "max_samples": 1,
                            "learning_rate": 2e-5,
                            "reference_learning_rate": 1e-4,
                            "loss_lambda_video": 1.0,
                            "loss_lambda_action": 4.0,
                            "save_every": 50,
                            "save_training_state": False,
                            "eval_every": 50,
                            "eval_fixed_index": 0,
                            "eval_at_start": True,
                            "eval_use_train_dataset": True,
                        },
                        "dataset_overfit": {
                            "task": "stage3_robocoin_memory_smoke",
                            "requires_memory_joint_inference": True,
                            "allowed_modes": ["joint_video_action", "video_only"],
                            "allowed_quality_tiers": ["A"],
                            "splits": ["train", "validation"],
                            "include_robot_supervision": True,
                            "skip_dit_load_from_pretrain": True,
                            "initial_checkpoint": str(self.root / "stage2.pt"),
                            "camera_rectification_config": str(self.rectification),
                            "sample_offset": 0,
                            "max_samples": None,
                            "max_samples_per_case": 3,
                            "sampling_strategy": "uniform",
                            "num_workers": 2,
                            "eval_fixed_indices": [5, 87, 180],
                        }
                    },
                }
            },
        }
        path = self.root / "tuning.yaml"
        path.write_text(yaml.safe_dump(config))
        return path

    def test_fastwam_builder_expands_environment_and_keeps_argv_structured(self) -> None:
        with patch.dict(os.environ, {"TEST_FASTWAM_REPO": str(self.repo)}):
            config = load_tuning_config(self._write_config())
        spec = build_fastwam_command(
            config,
            phase="stage3_finetune",
            output_dir=self.root / "run",
            steps=2,
            resume=None,
            gpus="3",
        )
        self.assertEqual(spec.env["CUDA_VISIBLE_DEVICES"], "3")
        self.assertIn("max_steps=2", spec.argv)
        self.assertIn("model.skip_dit_load_from_pretrain=true", spec.argv)
        self.assertTrue(any(value.startswith("resume=") for value in spec.argv))
        self.assertTrue(any(value.startswith("data.train.case_manifests=[") for value in spec.argv))

        checkpoint = self.root / "relative" / "checkpoint"
        checkpoint.mkdir(parents=True)
        relative_resume = Path(os.path.relpath(checkpoint, Path.cwd()))
        resumed = build_fastwam_command(
            config,
            phase="stage3_finetune",
            output_dir=self.root / "run",
            steps=2,
            resume=relative_resume,
            gpus="3",
        )
        self.assertIn(f"resume={relative_resume.resolve()}", resumed.argv)

    def test_fastwam_builder_uses_torchrun_for_multiple_gpus(self) -> None:
        with patch.dict(os.environ, {"TEST_FASTWAM_REPO": str(self.repo)}):
            config = load_tuning_config(self._write_config())

        spec = build_fastwam_command(
            config,
            phase="stage3_finetune",
            output_dir=self.root / "distributed-run",
            steps=1250,
            resume=None,
            gpus="0,2,4,6",
        )

        self.assertEqual(spec.env["CUDA_VISIBLE_DEVICES"], "0,2,4,6")
        self.assertEqual(
            spec.argv[:6],
            (
                str(self.python),
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=4",
                "scripts/train.py",
            ),
        )
        self.assertIn("++expected_world_size=4", spec.argv)
        self.assertIn("max_steps=1250", spec.argv)

    def test_shw5g_spec_rectification_derives_intrinsics_from_documented_fov(self) -> None:
        path = PROJECT_ROOT / "configs" / "cameras" / "tianji_shw5g_spec_fov_v1.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        profile = payload["profiles"]["tianji_shw5g_960x744_spec_equidistant"]
        matrix = profile["camera_matrix"]

        self.assertEqual(payload["provenance"]["native_output_pixels"], [2560, 1984])
        self.assertEqual(payload["provenance"]["dataset_pixels"], [960, 744])
        self.assertAlmostEqual(matrix[0][0], 479.5 / math.radians(65.0), places=10)
        self.assertAlmostEqual(matrix[1][1], 371.5 / math.radians(51.0), places=10)
        self.assertEqual(profile["distortion_coefficients"], [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(profile["virtual_camera"]["horizontal_fov_degrees"], 90.0)
        self.assertEqual(len(payload["bindings"]["source_key"]), 4)
        self.assertEqual(payload["unmatched_policy"], "error")

    def test_fastwam_builder_rejects_duplicate_gpu_ids(self) -> None:
        with patch.dict(os.environ, {"TEST_FASTWAM_REPO": str(self.repo)}):
            config = load_tuning_config(self._write_config())

        with self.assertRaisesRegex(TuningConfigError, "unique non-negative"):
            build_fastwam_command(
                config,
                phase="stage3_finetune",
                output_dir=self.root / "distributed-run",
                steps=2,
                resume=None,
                gpus="0,0",
            )

    def test_fastwam_overfit_phase_freezes_sample_and_evaluation(self) -> None:
        model_source = self.repo / "src" / "fastwam" / "models" / "wan22"
        model_source.mkdir(parents=True)
        (model_source / "memory_fastwam.py").write_text(
            "class MemoryFastWAM:\n"
            "    def infer_joint(self, memory_video_long, memory_video_mid, "
            "memory_video_short, memory_mask_long, memory_mask_mid, memory_mask_short):\n"
            "        pass\n"
            "    def infer(self, memory_video_long, memory_video_mid, memory_video_short, "
            "memory_mask_long, memory_mask_mid, memory_mask_short):\n"
            "        pass\n"
        )
        trainer_source = self.repo / "src" / "fastwam" / "trainer.py"
        trainer_source.parent.mkdir(parents=True, exist_ok=True)
        trainer_source.write_text(
            "eval_fixed_index = eval_at_start = memory_video_long = True\n"
        )
        with patch.dict(os.environ, {"TEST_FASTWAM_REPO": str(self.repo)}):
            config = load_tuning_config(self._write_config())
        spec = build_fastwam_command(
            config,
            phase="overfit",
            output_dir=self.root / "overfit-run",
            steps=300,
            resume=None,
            gpus="7",
        )

        self.assertIn("max_steps=300", spec.argv)
        self.assertIn("++data.train.sample_offset=1", spec.argv)
        self.assertIn("data.train.max_samples=1", spec.argv)
        self.assertIn("eval_fixed_index=0", spec.argv)
        self.assertIn("eval_at_start=true", spec.argv)
        self.assertIn("eval_use_train_dataset=true", spec.argv)
        self.assertIn("eval_every=50", spec.argv)
        self.assertIn("eval_seed=42", spec.argv)
        self.assertIn("save_every=50", spec.argv)
        self.assertIn("save_training_state=false", spec.argv)
        self.assertIn("learning_rate=2e-05", spec.argv)
        self.assertIn("reference_learning_rate=0.0001", spec.argv)
        self.assertIn("model.loss.lambda_video=1.0", spec.argv)
        self.assertIn("model.loss.lambda_action=4.0", spec.argv)

    def test_fastwam_overfit_rejects_runtime_without_memory_joint_inference(self) -> None:
        with patch.dict(os.environ, {"TEST_FASTWAM_REPO": str(self.repo)}):
            config = load_tuning_config(self._write_config())

        with self.assertRaisesRegex(TuningConfigError, "custom MemoryFastWAM"):
            build_fastwam_command(
                config,
                phase="overfit",
                output_dir=self.root / "overfit-run",
                steps=10,
                resume=None,
                gpus="0",
            )

    def test_fastwam_dataset_overfit_uses_all_splits_and_rectification(self) -> None:
        model_source = self.repo / "src" / "fastwam" / "models" / "wan22"
        model_source.mkdir(parents=True)
        (model_source / "memory_fastwam.py").write_text(
            "class MemoryFastWAM:\n"
            "    def infer_joint(self, memory_video_long, memory_video_mid, "
            "memory_video_short, memory_mask_long, memory_mask_mid, memory_mask_short):\n"
            "        pass\n"
            "    def infer(self, memory_video_long, memory_video_mid, memory_video_short, "
            "memory_mask_long, memory_mask_mid, memory_mask_short):\n"
            "        pass\n"
        )
        trainer_source = self.repo / "src" / "fastwam" / "trainer.py"
        trainer_source.parent.mkdir(parents=True, exist_ok=True)
        trainer_source.write_text(
            "eval_fixed_index = eval_at_start = memory_video_long = True\n"
        )
        with patch.dict(os.environ, {"TEST_FASTWAM_REPO": str(self.repo)}):
            config = load_tuning_config(self._write_config())

        spec = build_fastwam_command(
            config,
            phase="dataset_overfit",
            output_dir=self.root / "dataset-overfit-run",
            steps=1000,
            resume=None,
            gpus="6",
        )

        self.assertIn("data.train.max_samples=null", spec.argv)
        self.assertIn("++data.train.sample_offset=0", spec.argv)
        self.assertIn("++data.train.max_samples_per_case=3", spec.argv)
        self.assertIn("data.train.split=[train,validation]", spec.argv)
        self.assertIn(
            f"++data.train.camera_rectification_config={self.rectification.resolve()}",
            spec.argv,
        )
        self.assertIn("data.train.allowed_modes=[joint_video_action,video_only]", spec.argv)
        self.assertIn("sampling_strategy=uniform", spec.argv)
        self.assertIn("num_workers=2", spec.argv)
        self.assertIn("++eval_fixed_indices=[5,87,180]", spec.argv)

    def test_lingbot_builder_requires_complete_latents_unless_smoke_is_explicit(self) -> None:
        repo = self.root / "lingbot-va"
        repo.mkdir()
        model_root = self.root / "lingbot-model"
        model_root.mkdir()
        dataset = self.root / "lingbot-target"
        (dataset / "meta").mkdir(parents=True)
        jobs = []
        for episode in (0, 1):
            for camera in ("head", "left_wrist", "right_wrist"):
                output = f"latents/{camera}/episode_{episode:06d}.pth"
                jobs.append(
                    {
                        "episode_index": episode,
                        "start_frame": 0,
                        "end_frame": 80,
                        "camera_key": camera,
                        "output": output,
                    }
                )
                if episode == 0:
                    path = dataset / output
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("latent")
        (dataset / "meta" / "lingbot_va_latent_jobs.jsonl").write_text(
            "".join(json.dumps(job) + "\n" for job in jobs)
        )
        config = {
            "models": {
                "lingbot_va": {
                    "repo": str(repo),
                    "python": str(self.python),
                    "dataset_root": str(dataset),
                    "model_root": str(model_root),
                    "num_gpus": 1,
                }
            }
        }
        with self.assertRaisesRegex(TuningConfigError, "incomplete latents: 3/6"):
            build_lingbot_va_command(
                config,
                phase="finetune",
                output_dir=self.root / "run-lingbot",
                steps=1,
                resume=None,
                gpus="4",
            )

        config["models"]["lingbot_va"]["allow_partial_latents"] = True
        spec = build_lingbot_va_command(
            config,
            phase="finetune",
            output_dir=self.root / "run-lingbot",
            steps=1,
            resume=None,
            gpus="4",
        )
        self.assertEqual(spec.env["CUDA_VISIBLE_DEVICES"], "4")
        self.assertIn("--gradient-accumulation-steps", spec.argv)
        self.assertEqual(spec.argv[spec.argv.index("--batch-size") + 1], "1")

        config["models"]["lingbot_va"]["batch_size"] = 2
        with self.assertRaisesRegex(TuningConfigError, "positive window_frames"):
            build_lingbot_va_command(
                config,
                phase="finetune",
                output_dir=self.root / "run-lingbot-batch",
                steps=1,
                resume=None,
                gpus="4",
            )
        config["models"]["lingbot_va"].update(
            {"window_frames": 16, "samples_per_episode": 4}
        )
        batched = build_lingbot_va_command(
            config,
            phase="finetune",
            output_dir=self.root / "run-lingbot-batch",
            steps=1,
            resume=None,
            gpus="4",
        )
        self.assertEqual(batched.argv[batched.argv.index("--batch-size") + 1], "2")
        self.assertEqual(batched.argv[batched.argv.index("--window-frames") + 1], "16")

    def test_dreamzero_builder_validates_installed_profile(self) -> None:
        repo = self.root / "dreamzero"
        profile = (
            repo
            / "groot"
            / "vla"
            / "configs"
            / "data"
            / "dreamzero"
            / "xdof_relative.yaml"
        )
        profile.parent.mkdir(parents=True)
        profile.write_text("# fixture\n")
        dataset = self.root / "dreamzero-target"
        dataset.mkdir()
        wan = self.root / "wan"
        tokenizer = self.root / "tokenizer"
        pretrained = self.root / "dreamzero-model"
        for directory in (wan, tokenizer, pretrained):
            directory.mkdir()
        files = {}
        for key in ("text_encoder", "image_encoder", "vae"):
            files[key] = self.root / f"{key}.pth"
            files[key].write_text("weights")
        config = {
            "models": {
                "dreamzero": {
                    "repo": str(repo),
                    "python": str(self.python),
                    "dataset_root": str(dataset),
                    "data_profile": "dreamzero/xdof_relative",
                    "wan_model_root": str(wan),
                    "tokenizer_root": str(tokenizer),
                    "pretrained_model_root": str(pretrained),
                    "gpus": "0,1,2,3,4,5,6,7",
                    "num_gpus": 8,
                    "batch_size": 2,
                    "global_batch_size": 32,
                    "save_interval": 17,
                    "save_total_limit": 5,
                    "workers": 3,
                    **{key: str(value) for key, value in files.items()},
                }
            }
        }
        spec = build_dreamzero_command(
            config,
            phase="finetune",
            output_dir=self.root / "run-dreamzero",
            steps=2,
            resume=None,
            gpus="0,1,2,3,4,5,6,7",
        )
        self.assertEqual(spec.env["CUDA_VISIBLE_DEVICES"], "0,1,2,3,4,5,6,7")
        self.assertIn("data=dreamzero/xdof_relative", spec.argv)
        self.assertIn("max_steps=2", spec.argv)
        self.assertIn("per_device_train_batch_size=2", spec.argv)
        self.assertIn("global_batch_size=32", spec.argv)
        self.assertIn("save_steps=2", spec.argv)
        self.assertIn("save_total_limit=5", spec.argv)
        self.assertIn("dataloader_num_workers=3", spec.argv)
        self.assertIn("dataloader_persistent_workers=true", spec.argv)

        config["models"]["dreamzero"]["global_batch_size"] = 31
        with self.assertRaisesRegex(TuningConfigError, "positive multiple"):
            build_dreamzero_command(
                config,
                phase="finetune",
                output_dir=self.root / "run-dreamzero",
                steps=2,
                resume=None,
                gpus="0,1,2,3,4,5,6,7",
            )
        config["models"]["dreamzero"]["global_batch_size"] = 32

        profile.unlink()
        with self.assertRaisesRegex(TuningConfigError, "profile is not installed"):
            build_dreamzero_command(
                config,
                phase="finetune",
                output_dir=self.root / "run-dreamzero",
                steps=2,
                resume=None,
                gpus="0,1,2,3,4,5,6,7",
            )

    def test_dreamzero_wrapper_pins_explicit_checkpoint_and_skips_full_save(self) -> None:
        repo = self.root / "dreamzero-wrapper-fixture"
        package = repo / "groot" / "vla" / "experiment"
        package.mkdir(parents=True)
        for init in (
            repo / "groot" / "__init__.py",
            repo / "groot" / "vla" / "__init__.py",
            package / "__init__.py",
        ):
            init.write_text("")
        (package / "base.py").write_text(
            "def safe_save_model_for_hf_trainer(*, trainer, output_dir):\n"
            "    raise RuntimeError('generic saver was not patched')\n\n"
            "def get_checkpoint_path(output_dir, checkpoint_prefix='checkpoint'):\n"
            "    return None, False\n\n"
            "def get_last_checkpoint(output_dir):\n"
            "    return 'wrong-checkpoint'\n"
        )
        (package / "experiment.py").write_text(
            "import json\n"
            "import os\n"
            "from pathlib import Path\n"
            "import groot.vla.experiment.base as base\n\n"
            "selected, should_continue = base.get_checkpoint_path('ignored')\n"
            "trainer_selected = base.get_last_checkpoint('ignored')\n"
            "base.safe_save_model_for_hf_trainer(trainer=None, output_dir='ignored')\n"
            "Path(os.environ['WRAPPER_RESULT']).write_text(json.dumps({\n"
            "    'selected': selected,\n"
            "    'trainer_selected': trainer_selected,\n"
            "    'should_continue': should_continue,\n"
            "}))\n"
        )
        checkpoint = self.root / "checkpoint-1"
        checkpoint.mkdir()
        result_path = self.root / "wrapper-result.json"
        environment = os.environ.copy()
        environment["WRAPPER_RESULT"] = str(result_path)
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_dreamzero_training.py"),
                "--dreamzero-repo",
                str(repo),
                "--resume-from",
                str(checkpoint),
                "--",
                "save_lora_only=true",
            ],
            check=True,
            env=environment,
            text=True,
            capture_output=True,
        )
        result = json.loads(result_path.read_text())
        self.assertTrue(result["should_continue"])
        self.assertEqual(result["selected"], str(checkpoint.resolve()))
        self.assertEqual(result["trainer_selected"], str(checkpoint.resolve()))

    def test_missing_environment_variable_is_rejected(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(TuningConfigError, "TEST_FASTWAM_REPO"):
                load_tuning_config(self._write_config())


if __name__ == "__main__":
    unittest.main()
