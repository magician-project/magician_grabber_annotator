#!/usr/bin/python3

""" 
Author : "Ammar Qammaz"
Copyright : "2025 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH" 
"""

import wx
import os
import sys
import shutil
import re

def do_merge(src_path,dst_path):
   if (os.path.exists(src_path) and os.path.exists(dst_path)):
        # Collect existing files in target to determine max index
        existing_files = os.listdir(dst_path)
        max_num = 0
        pattern = re.compile(r"(\d+)\.png$")
        for f in existing_files:
            m = pattern.search(f)
            if m:
                max_num = max(max_num, int(m.group(1)))

        next_num = max_num + 1

        # Move files
        for fname in os.listdir(src_path):
            if not fname.lower().endswith(".png"):
                continue
            new_name = f"{dst_path}_{next_num}.png"
            while os.path.exists(os.path.join(dst_path, new_name)):
                next_num += 1
                new_name = f"{dst_path}_{next_num}.png"

            shutil.move(os.path.join(src_path, fname), os.path.join(dst_path, new_name))
            next_num += 1

        # Remove source directory if empty
        if not os.listdir(src_path):
            os.rmdir(src_path)


class MergeFrame(wx.Frame):
    def __init__(self, root_path):
        super().__init__(None, title="Class Merger", size=(500, 200))
        self.root_path = root_path
        self.class_dirs = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]

        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        # Source choice
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        hbox1.Add(wx.StaticText(panel, label="Source:"), flag=wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, border=8)
        self.source_choice = wx.Choice(panel, choices=self.class_dirs)
        hbox1.Add(self.source_choice, proportion=1)
        vbox.Add(hbox1, flag=wx.EXPAND | wx.ALL, border=10)

        # Target choice
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        hbox2.Add(wx.StaticText(panel, label="Target:"), flag=wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, border=8)
        self.target_choice = wx.Choice(panel, choices=self.class_dirs)
        hbox2.Add(self.target_choice, proportion=1)
        vbox.Add(hbox2, flag=wx.EXPAND | wx.ALL, border=10)

        # Merge button
        merge_btn = wx.Button(panel, label="Merge")
        merge_btn.Bind(wx.EVT_BUTTON, self.on_merge)
        vbox.Add(merge_btn, flag=wx.ALL | wx.ALIGN_CENTER, border=10)

        panel.SetSizer(vbox)

        # Initial populate
        self.refresh_choices()

    def refresh_choices(self):
        """Refresh source and target dropdown menus."""
        self.class_dirs = [d for d in os.listdir(self.root_path) if os.path.isdir(os.path.join(self.root_path, d))]
        self.source_choice.SetItems(self.class_dirs)
        self.target_choice.SetItems(self.class_dirs)

    def on_merge(self, event):
        src = self.source_choice.GetStringSelection()
        dst = self.target_choice.GetStringSelection()
        if not src or not dst:
            wx.MessageBox("Please select both source and target.", "Error", wx.OK | wx.ICON_ERROR)
            return
        if src == dst:
            wx.MessageBox("Source and target cannot be the same.", "Error", wx.OK | wx.ICON_ERROR)
            return

        src_path = os.path.join(self.root_path, src)
        dst_path = os.path.join(self.root_path, dst)

        do_merge(src_path, dst_path)

        wx.MessageBox(f"Merged '{src}' into '{dst}' successfully.", "Success", wx.OK | wx.ICON_INFORMATION)


        # Refresh dropdowns after merge
        self.refresh_choices()


def main():
    if len(sys.argv) < 2:
        print("Usage: python merge_classes.py <path>")
        sys.exit(1)

    if len(sys.argv) > 3:
        print("Usage: python merge_classes.py <path>")
        print("OR python merge_classes.py <source_path> <target_path>")
        sys.exit(1)

    if len(sys.argv) == 3:
        source_path = sys.argv[1]
        target_path = sys.argv[2]
        do_merge(source_path,target_path)
        sys.exit(0)


    root_path = sys.argv[1]
    if not os.path.isdir(root_path):
        print(f"Invalid path: {root_path}")
        sys.exit(1)

    app = wx.App(False)
    frame = MergeFrame(root_path)
    frame.Show()
    app.MainLoop()


if __name__ == "__main__":
    main()

