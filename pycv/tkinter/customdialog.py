
import tkinter as tk

class CustomDialog(tk.Toplevel):
    def __init__(self, parent, options, title, text="Select an option"):
        super().__init__(parent)
        self.parent = parent
        self.result = None

        self.title(title)
        self.geometry("300x150")
        self.resizable(False, False)

        # Make it modal
        self.transient(parent)
        self.grab_set()

        tk.Label(self, text=text, font=("Arial", 12)).pack(pady=10)

        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=10)
        for i, option in enumerate(options):
            tk.Button(btn_frame, text=option, width=10,
                      command=lambda i=i: self._set_result(i)).grid(row=0, column=i, padx=5)

        # Wait until the window is closed
        self.wait_window(self)

    def _set_result(self, value):
        self.result = value
        self.destroy()
