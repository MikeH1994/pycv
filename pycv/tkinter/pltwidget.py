import tkinter as tk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk
)
from matplotlib.widgets import RectangleSelector, LassoSelector
from matplotlib.path import Path


class PLTWidget(tk.Frame):
    def __init__(self, parent, x=np.array([]), y=np.array([]), xlabel="", ylabel="", title="",
                 forwards_button_callback=None, back_button_callback=None, additional_reset_callback = None,
                 additional_delete_callback=None, min_width=650):
        self.min_width = min_width
        super().__init__(parent)
        self.additional_reset_callback = additional_reset_callback
        self.addition_delete_callback = additional_delete_callback

        self.fig = Figure(figsize=(5, 4))
        self.ax = self.fig.add_subplot(111)

        self.x = np.asarray(x)
        self.y = np.asarray(y)

        self.points_mask = np.ones(len(self.x), dtype=bool)   # visible points
        self.selected = np.zeros(len(self.x), dtype=bool)     # selected points

        self.points = np.column_stack((self.x, self.y))

        self.scatter = self.ax.scatter(self.x, self.y, c='blue')
        self.line, = self.ax.plot(self.x, self.y, linestyle='--', color='gray')

        # ----------------------------
        # CANVAS
        # ----------------------------
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.configure(width=min_width)

        # ✅ Key binding for delete
        self.canvas.mpl_connect("key_press_event", self.on_key_press)

        # ----------------------------
        # TOOLBAR
        # ----------------------------
        self.toolbar = CustomToolbar(self.canvas, self, self, self.reset_selection,
                                          forwards_button_callback, back_button_callback)
        self.toolbar.update()

        self.rect_selector = None
        self.lasso_selector = None

        self.set_title(title)
        self.set_x_label(xlabel)
        self.set_y_label(ylabel)

    def update_plot(self):
        """Update scatter + line based on visibility + selection."""

        n = len(self.x)

        # --- Colors ---
        colors = np.array(['blue'] * n, dtype=object)
        colors[self.selected & self.points_mask] = 'red'

        # --- Sizes ---
        sizes = np.where(self.points_mask, 20, 0)

        self.scatter.set_offsets(self.points)
        self.scatter.set_color(colors)
        self.scatter.set_sizes(sizes)

        # --- Update line (only visible points) ---
        visible_x = self.x[self.points_mask]
        visible_y = self.y[self.points_mask]
        self.line.set_data(visible_x, visible_y)

        self.canvas.draw_idle()

    def on_key_press(self, event):
        if event.key == "delete":
            # Hide selected points
            self.points_mask[self.selected] = False
            self.selected[:] = False
            self.update_plot()
            if self.addition_delete_callback is not None:
                self.addition_delete_callback()

    def set_data(self, x, y):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.points = np.column_stack((self.x, self.y))

        self.points_mask = np.ones(len(self.x), dtype=bool)
        self.selected = np.zeros(len(self.x), dtype=bool)

        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.set_autoscale_on(False)

        self.update_plot()

    def set_title(self, title):
        self.ax.set_title(title)
        self.canvas.draw_idle()

    def set_x_label(self, xlabel):
        self.ax.set_xlabel(xlabel)
        self.canvas.draw_idle()

    def set_y_label(self, ylabel):
        self.ax.set_ylabel(ylabel)
        self.canvas.draw_idle()

    def deactivate_all(self):
        if self.rect_selector:
            self.rect_selector.set_active(False)
            self.rect_selector = None

        if self.lasso_selector:
            self.lasso_selector.disconnect_events()
            self.lasso_selector = None

    def activate_rectangle(self):
        def on_select(eclick, erelease):
            if eclick.xdata is None or erelease.xdata is None:
                return

            x_min, x_max = sorted([eclick.xdata, erelease.xdata])
            y_min, y_max = sorted([eclick.ydata, erelease.ydata])

            self.selected = (
                (self.x >= x_min) & (self.x <= x_max) &
                (self.y >= y_min) & (self.y <= y_max)
            )

            self.update_plot()

        self.rect_selector = RectangleSelector(
            self.ax,
            on_select,
            useblit=False,
            interactive=False
        )

    def activate_lasso(self):
        def on_lasso(verts):
            path = Path(verts)
            self.selected = path.contains_points(self.points)
            self.update_plot()

        self.lasso_selector = LassoSelector(self.ax, on_lasso)

    # ==================================================
    # ✅ RESET + STATE ACCESS
    # ==================================================
    def reset_selection(self):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        self.points_mask[:] = True
        self.selected[:] = False

        self.update_plot()

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

        if self.additional_reset_callback is not None:
            self.additional_reset_callback()

    def get_point_visibility(self):
        """Return boolean array of point visibility."""
        return self.points_mask.copy()

    def set_point_visibility(self, i, state):
        self.points_mask[i] = state
        self.update_plot()


class CustomToolbar(NavigationToolbar2Tk):
    def __init__(self, canvas, window, widget, reset_button_callback=None, forwards_button_callback = None, backwards_button_callback = None):
        super().__init__(canvas, window)
        self.widget = widget

        self.tool_var = tk.StringVar(value="none")
        self.last_tool = "none"

        # --- ROI Buttons ---
        self.rect_btn = tk.Radiobutton(
            self, text="Rect",
            variable=self.tool_var, value="rect",
            indicatoron=False,
            command=self.on_tool_change,
            width=8
        )
        self.rect_btn.pack(side=tk.LEFT)

        self.lasso_btn = tk.Radiobutton(
            self, text="Lasso",
            variable=self.tool_var, value="lasso",
            indicatoron=False,
            command=self.on_tool_change,
            width=8
        )
        self.lasso_btn.pack(side=tk.LEFT)
        self.custom_back_btn = tk.Button(self, text=" ◀ ", command=forwards_button_callback)
        self.custom_back_btn.pack(side=tk.LEFT)
        self.custom_forward_btn = tk.Button(self, text=" ▶ ", command=backwards_button_callback)
        self.custom_forward_btn.pack(side=tk.LEFT)
        self.custom_reset_btn = tk.Button(self, text=" ⟳ ", command=reset_button_callback)
        self.custom_reset_btn.pack(side=tk.LEFT)


    # ----------------------------
    # Turn OFF matplotlib tools
    # ----------------------------
    def deactivate_mpl_tools(self):
        if self.mode == "zoom rect":
            super().zoom()
        elif self.mode == "pan/zoom":
            super().pan()

    # ----------------------------
    # ROI switching
    # ----------------------------
    def on_tool_change(self):
        selected = self.tool_var.get()

        # Toggle OFF
        if selected == self.last_tool:
            self.tool_var.set("none")
            self.widget.deactivate_all()
            self.last_tool = "none"
            return

        # Turn off MPL tools
        self.deactivate_mpl_tools()

        # Activate ROI tools
        self.widget.deactivate_all()

        if selected == "rect":
            self.widget.activate_rectangle()
        elif selected == "lasso":
            self.widget.activate_lasso()

        self.last_tool = selected

    def deactivate_custom_tools(self):
        if self.tool_var.get() != "none":
            self.tool_var.set("none")
            self.widget.deactivate_all()
            self.last_tool = "none"

    # ----------------------------
    # Override MPL buttons
    # ----------------------------
    def zoom(self, *args):
        self.deactivate_custom_tools()
        super().zoom(*args)

    def pan(self, *args):
        self.deactivate_custom_tools()
        super().pan(*args)

    def home(self, *args):
        self.deactivate_custom_tools()
        super().home(*args)

    def back(self, *args):
        self.deactivate_custom_tools()
        super().back(*args)

    def forward(self, *args):
        self.deactivate_custom_tools()
        super().forward(*args)