from __future__ import annotations
import tkinter
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import RectangleSelector
import tkinter as tk
from typing import Tuple
from numpy.typing import NDArray
import numpy as np
from PIL import ImageTk, Image
import threading
from typing import List, Tuple, Callable
import time
import cv2
from tkinter import simpledialog, messagebox, IntVar, DoubleVar, Label
from tkinter import filedialog
import math
import matplotlib.pyplot as plt


class BaseGUI:
    def __init__(self, window: tkinter.Tk):
        self.window = window
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        self.menu = tkinter.Menu(self.window)
        self.window.config(menu=self.menu)


    def on_close(self):
        self.window.quit()
        self.window.destroy()

    # noinspection PyMethodMayBeStatic
    def add_listbox(self, master, row=0, column=0, callback: Callable=None, **kwargs):
        frame = tkinter.Frame(master, **kwargs)
        frame.grid(row=row, column=column)
        listbox = tkinter.Listbox(frame)
        listbox.grid(column=0, row=0,  sticky="nsew")

        # add vertical scrollbar
        y_scrollbar = tkinter.Scrollbar(frame)
        y_scrollbar.grid(column=1, row=0, sticky="nsew")
        listbox.config(yscrollcommand=y_scrollbar.set)
        y_scrollbar.config(command=listbox.yview)

        # Create the horizontal scrollbar
        x_scrollbar = tk.Scrollbar(frame, orient=tk.HORIZONTAL, command=listbox.xview)
        x_scrollbar.grid(column=0, row=1, sticky="nsew")
        listbox.config(xscrollcommand=x_scrollbar.set)
        x_scrollbar.config(command=listbox.xview)

        if callback is not None:
            listbox.bind("<<ListboxSelect>>", callback)

        return listbox

    # noinspection PyMethodMayBeStatic
    def insert_element_into_listbox(self, listbox, name):
        listbox.insert("end", name)

    # noinspection PyMethodMayBeStatic
    def get_listbox_selection(self, listbox):
        return listbox.curselection()

    # noinspection PyMethodMayBeStatic
    def add_matplotlib_plot(self, parent, row=0, column=0):
        frame = self.add_frame(parent, row, column)

        # Create the figure and axis
        fig, ax = plt.subplots()
        fig.set_size_inches(12,8)
        plot, = ax.plot([], [])  # empty line to update later

        # Embed the canvas in the Tkinter frame
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas.draw()

        # Add the Matplotlib navigation toolbar
        toolbar = NavigationToolbar2Tk(canvas, frame)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)

        return fig, ax, plot, canvas

    def add_rectangle_selector_to_matplotlib_figure(self, ax, callback: Callable):
        toggle_selector = RectangleSelector(
            ax, callback,
            useblit=True,
            button=[1],  # Left mouse button
            minspanx=5, minspany=5,
            interactive=True
        )

    # noinspection PyMethodMayBeStatic
    def update_matplotlib_plot(self, ax, canvas):
        ax.relim()        # Recalculate limits
        ax.autoscale_view()  # Rescale the view
        canvas.draw()     # Redraw the canvas

    def add_numerical_entry(self, parent, row, column, default_value: int, callback=None):
        def validate_positive_integer(new_value):
            if new_value == "":
                return True
            return new_value.isdigit() and int(new_value) > 0

        entry_var = tk.StringVar(value=str(default_value))
        if callback is not None:
            entry_var.trace_add("write", callback)  # Callback on value change

        vcmd = (parent.register(validate_positive_integer), "%P")
        entry = tk.Entry(parent, validate="key", validatecommand=vcmd, textvariable=entry_var)
        entry.grid(row=row, column=column, sticky="nesw")
        return entry, entry_var


    def add_label(self, parent, text, row=0, column=0):
        label = Label(parent, text=text)
        label.grid(row=row, column=column, sticky="nsew")
        return label

    # noinspection PyMethodMayBeStatic
    def add_frame(self, parent, row=0, column=0, **kwargs):
        frame = tkinter.Frame(parent, **kwargs)
        if row is not None and column is not None:
            frame.grid(row=row, column=column, sticky="nsew")
        return frame

    # noinspection PyMethodMayBeStatic
    def add_menu_cascade(self, cascade_label, commands: List[Tuple[str, Callable | None]]):
        submenu = tkinter.Menu(self.menu, tearoff=False)
        for command_label, command in commands:
            submenu.add_command(label=command_label, command=command)
        self.menu.add_cascade(menu=submenu, label=cascade_label)

    # noinspection PyMethodMayBeStatic
    def get_tab_by_name(self, notebook, tab_name: str):
        for tab_id in notebook.tabs():
            if notebook.tab(tab_id, 'text') == tab_name:
                return tab_id
        return None

    # noinspection PyMethodMayBeStatic
    def add_notebook(self, parent, tab_names: List[str], row, column, **kwargs):
        tabs = ttk.Notebook(parent)
        tab_frames = {}
        for name in tab_names:
            frame = tkinter.Frame(tabs, **kwargs)
            tabs.add(frame, text=name)
            tab_frames[name] = frame
        tabs.grid(column=column, row=row, sticky="nsew")
        return tabs, tab_frames

    # noinspection PyMethodMayBeStatic
    def add_scrollbar(self, tab, from_, to_, resolution, text, default_value=None, callback=None):
        _, rows = tab.grid_size()
        row = rows + 1
        col = 0
        if isinstance(from_, float) and isinstance(to_, float) and isinstance(resolution, float):
            dst_var = DoubleVar()
        elif isinstance(from_, int) and isinstance(to_, int) and isinstance(resolution, int):
            dst_var = IntVar()
        else:
            raise Exception("scalebar and resolution must all be of same type")

        label = tkinter.Label(master=tab, text=text)
        label.grid(row=row, column=col, sticky="nsew")
        scrollbar = tkinter.Scale(tab, from_=from_, to=to_, orient=tkinter.HORIZONTAL,
                                  resolution=resolution, variable=dst_var)
        scrollbar.grid(row=row + 1, column=col, sticky="nsew")
        if default_value is not None:
            scrollbar.set(default_value)
        if callback is not None:
            scrollbar.bind("<ButtonRelease-1>", callback)

        return scrollbar, dst_var

    # noinspection PyMethodMayBeStatic
    def add_combobox(self, root, values, name):
        _, rows = root.grid_size()
        row = rows + 1
        col = 0
        label = tkinter.Label(master=root, text=name)
        label.grid(row=row, column=col, sticky="nsew")
        cb = ttk.Combobox(root, values=values, state="readonly")
        cb.set(values[0])
        cb.grid(row=row+1, column=col, sticky="nsew")
        return cb

    # noinspection PyMethodMayBeStatic
    def add_checkbox(self, tab, text, callback=None, default_value=None):
        _, rows = tab.grid_size()
        row = rows + 1
        col = 0
        is_checked = IntVar()
        checkbutton = tkinter.Checkbutton(tab, text=text, command=callback, variable=is_checked)
        # Setting options for the Checkbutton
        checkbutton.grid(row=row, column=col, sticky="nsew")
        if default_value is not None:
            is_checked.set(int(default_value))
        return checkbutton, is_checked
