import os # <-- NEW IMPORT
import tempfile
from typing import Optional

import numpy as np
from moviepy.editor import ImageSequenceClip, AudioFileClip
from scipy.io import wavfile


def save_video(
    output_path: str,
    video_numpy: np.ndarray,
    audio_data: Optional[np.ndarray] = None, # Renamed to audio_data for clarity
    temp_audio_path: Optional[str] = None, # <-- NEW ARGUMENT
    sample_rate: int = 16000,
    fps: int = 24,
) -> str:
    """
    Combine a sequence of video frames with an optional audio track and save as an MP4.

    Args:
        output_path (str): Path to the output MP4 file.
        video_numpy (np.ndarray): Numpy array of frames. Shape (C, F, H, W).
                                  Values can be in range [-1, 1] or [0, 255].
        audio_data (Optional[np.ndarray]): 1D or 2D numpy array of audio samples, range [-1, 1].
        temp_audio_path (Optional[str]): Path to save the intermediate WAV file. 
                                         If None, uses system temp (original behavior). 
                                         <-- NEW DESCRIPTION
        sample_rate (int): Sample rate of the audio in Hz. Defaults to 16000.
        fps (int): Frames per second for the video. Defaults to 24.

    Returns:
        str: Path to the saved MP4 file.
    """

    # Validate inputs
    assert isinstance(video_numpy, np.ndarray), "video_numpy must be a numpy array"
    assert video_numpy.ndim == 4, "video_numpy must have shape (C, F, H, W)"
    assert video_numpy.shape[0] in {1, 3}, "video_numpy must have 1 or 3 channels"

    if audio_data is not None:
        assert isinstance(audio_data, np.ndarray), "audio_data must be a numpy array"
        assert np.abs(audio_data).max() <= 1.0, "audio_data values must be in range [-1, 1]"

    # Reorder dimensions: (C, F, H, W) -> (F, H, W, C)
    video_numpy = video_numpy.transpose(1, 2, 3, 0)

    # Normalize frames if values are in [-1, 1]
    if video_numpy.max() <= 1.0:
        video_numpy = np.clip(video_numpy, -1, 1)
        video_numpy = ((video_numpy + 1) / 2 * 255).astype(np.uint8)
    else:
        video_numpy = video_numpy.astype(np.uint8)

    # Convert numpy array to a list of frames
    frames = list(video_numpy)

    # Create video clip
    clip = ImageSequenceClip(frames, fps=fps)
    
    # Define audio source path and cleanup flag
    audio_source_path = None
    should_cleanup_audio = False

    # Add audio if provided
    if audio_data is not None:
        
        # --- MODIFIED AUDIO HANDLING LOGIC ---
        if temp_audio_path:
            # Use the specified path (your local 'Temp' folder)
            audio_source_path = temp_audio_path
            should_cleanup_audio = True
        else:
            # Fallback to system temp directory (original behavior)
            # We use NamedTemporaryFile just to get the name, and then write to it
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_audio_path = temp_file.name
            temp_file.close()
            audio_source_path = temp_audio_path
            should_cleanup_audio = True # Ensure cleanup for the system temp file

        # Write the audio data to the determined path
        wavfile.write(
            audio_source_path,
            sample_rate,
            (audio_data * 32767).astype(np.int16),
        )

        # Create audio clip and set it on the video clip
        audio_clip = AudioFileClip(audio_source_path)
        final_clip = clip.set_audio(audio_clip)
    else:
        final_clip = clip

    # Write final video to disk
    final_clip.write_videofile(
        output_path, codec="libx264", audio_codec="aac", fps=fps, verbose=False, logger=None
    )
    final_clip.close()

    # Clean up the temporary WAV file ONLY if we created it internally
    if should_cleanup_audio and audio_source_path and os.path.exists(audio_source_path):
        os.remove(audio_source_path)
        
    return output_path