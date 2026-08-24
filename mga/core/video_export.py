#!/usr/bin/python3

"""
Author : "Ammar Qammaz"
Copyright : "2025 Foundation of Research and Technology, Computer Science Department Greece"
License : "FORTH"

Tools -> Make Video — extracted from mga/wx_annotator.py (Stage 2 of its
refactor). Renders every frame (left + right side-by-side) to JPEGs in a temp
directory, then encodes with ffmpeg. `app` is the PhotoCtrl instance: the
render loop walks frames through the GUI (wx.CallAfter per frame) because the
bitmaps are what the user sees, then a worker thread drives the pacing.
"""

import os
import wx
import cv2
import numpy as np
import subprocess
import threading
import tempfile


def export_dataset_video(app):
    """Render every frame (left + right side-by-side) to JPEGs then encode
    with ffmpeg (the body of wx_annotator.onMakeVideo)."""
    if not app.filePathIsDirectory:
        wx.MessageBox("Please open a directory first.", "Make Video", wx.OK | wx.ICON_INFORMATION)
        return

    total = app.folderStreamer.max()
    if total == 0:
        wx.MessageBox("No frames found.", "Make Video", wx.OK | wx.ICON_WARNING)
        return

    # Ask for output path
    with wx.FileDialog(app.frame, "Save video as", wildcard="MP4 files (*.mp4)|*.mp4",
                       style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as fd:
        if fd.ShowModal() != wx.ID_OK:
            return
        out_path = fd.GetPath()
    if out_path.endswith(".mp4"):
        out_path = out_path[:-4]

    # Work in a temp directory so frame JPEGs don't clutter the dataset
    tmp_dir = tempfile.mkdtemp(prefix="wxAnnotator_video_")

    dlg = wx.ProgressDialog(
        "Making Video", "Rendering frames…",
        maximum=total, parent=app.frame,
        style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_CAN_ABORT | wx.PD_ELAPSED_TIME)

    def _worker():
        saved_ui = app.scrollBar.GetValue()
        aborted  = False
        done_evt = threading.Event()
        cont_box = [True]

        def _bmp_to_cv(bmp):
            if not (bmp and bmp.IsOk()):
                return None
            img = bmp.ConvertToImage()
            arr = np.frombuffer(img.GetData(), dtype=np.uint8)
            arr = arr.reshape(img.GetHeight(), img.GetWidth(), 3)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        for i in range(total):
            done_evt.clear()

            def _gui_step(fi=i):
                # Update progress; check for user abort
                cont_box[0], _ = dlg.Update(fi, f"Frame {fi+1}/{total}")

                # Advance frame and force redraw
                app.gotoFrameUI(fi)
                app.panel.Update()
                wx.GetApp().Yield(True)

                left_cv  = _bmp_to_cv(app.imageCtrl.GetBitmap())
                right_cv = _bmp_to_cv(app.secondaryImageCtrl.GetBitmap())

                if left_cv is not None and right_cv is not None:
                    h = max(left_cv.shape[0], right_cv.shape[0])
                    def _pad(img, th):
                        ph = th - img.shape[0]
                        return np.pad(img, ((0, ph), (0, 0), (0, 0))) if ph > 0 else img
                    frame = np.concatenate([_pad(left_cv, h), _pad(right_cv, h)], axis=1)
                elif left_cv is not None:
                    frame = left_cv
                elif right_cv is not None:
                    frame = right_cv
                else:
                    done_evt.set()
                    return

                fname = os.path.join(tmp_dir, f"colorFrame_0_{fi:05d}.jpg")
                cv2.imwrite(fname, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                done_evt.set()

            wx.CallAfter(_gui_step)
            done_evt.wait(timeout=60)   # wait for GUI thread to finish this frame

            if not cont_box[0]:
                aborted = True
                break

        def _finalize():
            dlg.Destroy()

            if aborted:
                wx.MessageBox("Rendering aborted.", "Make Video", wx.OK | wx.ICON_INFORMATION)
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return

            # Restore original frame
            app.gotoFrameUI(saved_ui)

            # Encode with ffmpeg
            enc_dlg = wx.ProgressDialog("Encoding", "Running ffmpeg…", maximum=1,
                                         parent=app.frame, style=wx.PD_APP_MODAL)
            enc_dlg.Pulse()

            ret = subprocess.run([
                "ffmpeg", "-nostdin", "-framerate", "25",
                "-i", os.path.join(tmp_dir, "colorFrame_0_%05d.jpg"),
                "-vf", "scale=-2:720", "-y", "-r", str(app.metadata.get("frameRate",23)),
                "-pix_fmt", "yuv420p", "-threads", "8",
                f"{out_path}_lastRun3DHiRes.mp4",
            ], check=False)

            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
            enc_dlg.Destroy()

            if ret.returncode == 0:
                wx.MessageBox(f"Video saved:\n{out_path}_lastRun3DHiRes.mp4",
                              "Make Video", wx.OK | wx.ICON_INFORMATION)
            else:
                wx.MessageBox(
                    f"ffmpeg exited with code {ret.returncode}.\n"
                    "Is ffmpeg installed and on PATH?",
                    "Make Video", wx.OK | wx.ICON_ERROR)

        wx.CallAfter(_finalize)

    threading.Thread(target=_worker, daemon=True).start()
