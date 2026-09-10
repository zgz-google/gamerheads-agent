# Copyright 2026 Google LLC
from pathlib import Path

import pytest

from app.video_tool import create_mp4_video


@pytest.mark.asyncio
async def test_create_mp4_video_generation(tmp_path):
    output_name = "test_run.mp4"
    result = await create_mp4_video(
        title="Test Clip",
        description="Unit test clip",
        duration_seconds=2,
        theme="gaming",
        output_filename=output_name,
    )
    assert "Successfully generated MP4 video" in result
    assert "720x1280" in result
    assert "9:16 Vertical" in result
    output_file = Path.cwd() / "outputs" / output_name
    assert output_file.exists()
    assert output_file.stat().st_size > 0
    # Clean up test output
    output_file.unlink()


@pytest.mark.asyncio
async def test_create_mp4_video_padded(tmp_path):
    output_name = "test_padded_run.mp4"
    result = await create_mp4_video(
        title="Padded Clip",
        description="Padded test clip",
        duration_seconds=2,
        theme="cyberpunk",
        target_size_mb=2.0,
        output_filename=output_name,
    )
    assert "Successfully generated MP4 video" in result
    assert "2.0 MB" in result
    output_file = Path.cwd() / "outputs" / output_name
    assert output_file.exists()
    assert output_file.stat().st_size >= 2 * 1024 * 1024
    output_file.unlink()
