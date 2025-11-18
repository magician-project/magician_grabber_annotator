import wx

ID_NAMES = [
    'ID_ABORT','ID_ABOUT','ID_ADD','ID_ANY','ID_APPLY','ID_BACKWARD','ID_BOLD',
    'ID_CANCEL','ID_CLEAR','ID_CLOSE','ID_CLOSE_ALL','ID_CONTEXT_HELP','ID_COPY',
    'ID_CUT','ID_DEFAULT','ID_DELETE','ID_DOWN','ID_DUPLICATE','ID_EDIT','ID_EXIT',
    'ID_FILE','ID_FILE1','ID_FILE2','ID_FILE3','ID_FILE4','ID_FILE5','ID_FILE6',
    'ID_FILE7','ID_FILE8','ID_FILE9','ID_FIND','ID_FORWARD','ID_HELP',
    'ID_HELP_COMMANDS','ID_HELP_CONTENTS','ID_HELP_CONTEXT','ID_HELP_INDEX',
    'ID_HELP_PROCEDURES','ID_HELP_SEARCH','ID_HIGHEST','ID_HOME','ID_IGNORE',
    'ID_INDENT','ID_INDEX','ID_ITALIC','ID_JUSTIFY_CENTER','ID_JUSTIFY_FILL',
    'ID_JUSTIFY_LEFT','ID_JUSTIFY_RIGHT','ID_LOWEST','ID_MORE','ID_NEW','ID_NO',
    'ID_NONE','ID_NOTOALL','ID_OK','ID_OPEN','ID_PAGE_SETUP','ID_PASTE',
    'ID_PREFERENCES','ID_PREVIEW','ID_PREVIEW_CLOSE','ID_PREVIEW_FIRST',
    'ID_PREVIEW_GOTO','ID_PREVIEW_LAST','ID_PREVIEW_NEXT','ID_PREVIEW_PREVIOUS',
    'ID_PREVIEW_PRINT','ID_PREVIEW_ZOOM','ID_PRINT','ID_PRINT_SETUP','ID_PROPERTIES',
    'ID_REDO','ID_REFRESH','ID_REMOVE','ID_REPLACE','ID_REPLACE_ALL','ID_RESET',
    'ID_RETRY','ID_REVERT','ID_REVERT_TO_SAVED','ID_SAVE','ID_SAVEAS',
    'ID_SELECTALL','ID_SEPARATOR','ID_SETUP','ID_STATIC','ID_STOP','ID_UNDELETE',
    'ID_UNDERLINE','ID_UNDO','ID_UNINDENT','ID_UP','ID_VIEW_DETAILS',
    'ID_VIEW_LARGEICONS','ID_VIEW_LIST','ID_VIEW_SMALLICONS','ID_VIEW_SORTDATE',
    'ID_VIEW_SORTNAME','ID_VIEW_SORTSIZE','ID_VIEW_SORTTYPE','ID_YES','ID_YESTOALL',
    'ID_ZOOM_100','ID_ZOOM_FIT','ID_ZOOM_IN','ID_ZOOM_OUT'
]


class IDMenuFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="wx.ID_* Menu Picker", size=(800, 400))

        self.CreateStatusBar()
        self.SetStatusText("Pick any ID from the menus above")

        menubar = wx.MenuBar()

        # How many entries per dropdown menu
        GROUP_SIZE = 12

        # Build a reverse lookup from ID → name for printing later
        self.id_to_name = {}
        for name in ID_NAMES:
            if hasattr(wx, name):
                wx_id = getattr(wx, name)
                self.id_to_name[wx_id] = name

        # Create several menus with subsets of the IDs
        index = 0
        total = len(ID_NAMES)
        for start in range(0, total, GROUP_SIZE):
            end = min(start + GROUP_SIZE, total)
            group_names = ID_NAMES[start:end]

            menu = wx.Menu()
            for idx, name in enumerate(group_names):
                # Skip if this constant doesn't exist in this wx build
                if not hasattr(wx, name):
                    continue

                # Special handling for separator
                if name == 'ID_SEPARATOR':
                    menu.AppendSeparator()
                    continue

                wx_id = getattr(wx, name)
                # Use the constant name as the menu label, with '&' as accelerator marker
                label = name.replace('ID_', '&ID_')
                help_str = f"wx.{name} = {wx_id}"
                item = menu.Append(wx_id, label, help_str)
                self.Bind(wx.EVT_MENU, self.on_menu, item)

            title = f"IDs {start+1}-{end}"
            menubar.Append(menu, title)

        self.SetMenuBar(menubar)

        panel = wx.Panel(self)
        text = wx.StaticText(panel, label="Use the menu bar to pick an ID.")
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(text, 0, wx.ALL, 20)
        panel.SetSizer(sizer)

    def on_menu(self, event):
        wx_id = event.GetId()
        name = self.id_to_name.get(wx_id, f"(unknown, value={wx_id})")
        msg = f"Selected: {name}  →  wx ID value = {wx_id}"
        print(msg)
        self.SetStatusText(msg)


class App(wx.App):
    def OnInit(self):
        frame = IDMenuFrame()
        frame.Show()
        return True


if __name__ == "__main__":
    app = App(False)
    app.MainLoop()

