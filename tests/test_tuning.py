from __future__ import annotations

import json
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
            gpus="5",
        )
        self.assertEqual(spec.env["CUDA_VISIBLE_DEVICES"], "5")
        self.assertIn("data=dreamzero/xdof_relative", spec.argv)
        self.assertIn("max_steps=2", spec.argv)

        profile.unlink()
        with self.assertRaisesRegex(TuningConfigError, "profile is not installed"):
            build_dreamzero_command(
                config,
                phase="finetune",
                output_dir=self.root / "run-dreamzero",
                steps=2,
                resume=None,
                gpus="5",
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
