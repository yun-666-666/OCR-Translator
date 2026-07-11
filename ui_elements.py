import tkinter as tk
from tkinter import ttk
from logger import log_debug
from modern_ui import style_tk_canvas


_WHEEL_INPUT_WIDGET_CLASSES = frozenset({
    "TCombobox",
    "TSpinbox",
    "Spinbox",
    "TEntry",
    "Entry",
    "TScale",
    "Scale",
})


def _mousewheel_scroll_units(event):
    """Normalize Windows/macOS and X11 wheel events to canvas units."""
    button_number = getattr(event, "num", None)
    if button_number == 4:
        return -1
    if button_number == 5:
        return 1

    delta = int(getattr(event, "delta", 0) or 0)
    if delta == 0:
        return 0
    steps = max(1, abs(delta) // 120)
    return -steps if delta > 0 else steps


def _bind_wheel_input_guards(root_widget, handler):
    """Bind wheel routing before ttk class bindings on input descendants."""
    stack = list(root_widget.winfo_children())
    while stack:
        widget = stack.pop()
        stack.extend(widget.winfo_children())
        if widget.winfo_class() not in _WHEEL_INPUT_WIDGET_CLASSES:
            continue
        if getattr(widget, "_scroll_wheel_guard_installed", False):
            continue
        widget.bind("<MouseWheel>", handler, add="+")
        widget.bind("<Button-4>", handler, add="+")
        widget.bind("<Button-5>", handler, add="+")
        widget._scroll_wheel_guard_installed = True


def create_scrollable_tab(notebook, tab_name, protect_wheel_inputs=False):
    """Create a scrollable tab with conditional scrollbar visibility"""
    # Create the tab frame
    tab = ttk.Frame(notebook)
    notebook.add(tab, text=tab_name)
    
    # Create outer frame to hold canvas and scrollbar
    outer_frame = ttk.Frame(tab)
    outer_frame.pack(fill="both", expand=True)
    
    # Create canvas without border or highlight
    canvas = tk.Canvas(outer_frame, highlightthickness=0, borderwidth=0)
    style_tk_canvas(canvas)
    
    # Create scrollbar (not packed yet - will be shown conditionally)
    scrollbar = ttk.Scrollbar(outer_frame, orient="vertical", command=canvas.yview)
    
    # Create content frame for actual content
    content_frame = ttk.Frame(canvas)
    
    # Configure the canvas to use the scrollable frame
    canvas_window = canvas.create_window((0, 0), window=content_frame, anchor="nw")
    
    # Configure canvas to adjust with window size
    def configure_canvas(event):
        # Update the width of the canvas window to fill the canvas
        canvas.itemconfig(canvas_window, width=event.width)
        # Schedule scrollbar update after canvas resize
        canvas.after_idle(update_scrollbar)
    
    canvas.bind('<Configure>', configure_canvas)
    
    # Update scrollregion and scrollbar visibility
    def update_scrollbar():
        # Update scroll region first
        canvas.configure(scrollregion=canvas.bbox("all"))
        
        # Small delay to ensure layout is complete
        canvas.after(10, check_scrollbar_needed)
    
    def check_scrollbar_needed():
        try:
            # Get current dimensions
            canvas.update_idletasks()  # Ensure layout is complete
            content_height = content_frame.winfo_reqheight()
            canvas_height = canvas.winfo_height()
            
            # Show or hide scrollbar based on content height
            if content_height > canvas_height and canvas_height > 1:  # canvas_height > 1 ensures canvas is initialized
                if not scrollbar.winfo_ismapped():
                    scrollbar.pack(side="right", fill="y")
            else:
                if scrollbar.winfo_ismapped():
                    scrollbar.pack_forget()
        except tk.TclError:
            # Widget might be destroyed, ignore
            pass
    
    # Mouse wheel scrolling
    def _scroll_for_event(event):
        units = _mousewheel_scroll_units(event)
        if units and scrollbar.winfo_ismapped():
            canvas.yview_scroll(units, "units")

    def _on_mousewheel(event):
        _scroll_for_event(event)

    def _on_guarded_input_wheel(event):
        _scroll_for_event(event)
        return "break"

    input_guard_install_scheduled = False

    def schedule_input_wheel_guard():
        nonlocal input_guard_install_scheduled
        if not protect_wheel_inputs or input_guard_install_scheduled:
            return
        input_guard_install_scheduled = True

        def install_input_wheel_guard():
            nonlocal input_guard_install_scheduled
            input_guard_install_scheduled = False
            try:
                if content_frame.winfo_exists():
                    _bind_wheel_input_guards(
                        content_frame,
                        _on_guarded_input_wheel,
                    )
            except tk.TclError:
                pass

        canvas.after_idle(install_input_wheel_guard)

    # Re-scan after layout changes so inputs created later receive the same
    # Settings-page wheel behavior without duplicate callbacks.
    def _on_content_configure(event):
        update_scrollbar()
        schedule_input_wheel_guard()

    content_frame.bind('<Configure>', _on_content_configure)
    
    # Bind Enter/Leave to manage mousewheel event to avoid scroll conflicts
    def _on_enter(event):
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
    def _on_leave(event):
        canvas.unbind_all("<MouseWheel>")
    
    canvas.bind("<Enter>", _on_enter)
    canvas.bind("<Leave>", _on_leave)
    
    # Pack canvas first (scrollbar is packed conditionally)
    canvas.pack(side="left", fill="both", expand=True)
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # Ensure outer frame fills the tab
    outer_frame.pack(fill="both", expand=True)
    
    # Schedule initial scrollbar check after everything is set up
    canvas.after(100, update_scrollbar)
    schedule_input_wheel_guard()
    
    # Return the content frame where widgets should be placed
    return content_frame

class ResizableMovableFrame(tk.Toplevel):
    """A simple movable and resizable Toplevel window without decorations."""
    def __init__(self, parent, initial_geometry, bg_color, title="Movable Window"):
        super().__init__(parent)
        self.overrideredirect(True)  # No window decorations (title bar, borders)

        # Set initial geometry
        try:
            x1, y1, x2, y2 = map(int, initial_geometry) # Ensure integers
            width = max(x2 - x1, 100) # Ensure minimum practical width
            height = max(y2 - y1, 50) # Ensure minimum practical height
            self.geometry(f"{width}x{height}+{x1}+{y1}")
        except (ValueError, TypeError) as e:
            log_debug(f"Error setting initial geometry {initial_geometry}: {e}. Using default.")
            self.geometry("200x100+100+100") # Default fallback geometry

        # Semi-transparent background
        self.attributes("-alpha", 0.1) # Very transparent for source area visibility
        self.attributes("-topmost", True) # Keep on top

        # Set the background color for the main window
        self.configure(bg=bg_color)
        # Force the background color to be set properly
        self.update_idletasks()

        # Create a minimal title bar for moving - ISSUE 4 FIX: Set proper cursor
        self.title_bar = tk.Frame(self, bg=bg_color, height=5, cursor="fleur")
        self.title_bar.pack(fill=tk.X, side=tk.TOP)

        # Close button (very small)
        self.close_button = tk.Button(self.title_bar, text="×", bg=bg_color, fg="white",
                                     command=self.destroy, relief="flat", highlightthickness=0,
                                     font=("Arial", 6), width=1, height=1, bd=0, padx=0, pady=0)
        self.close_button.pack(side=tk.RIGHT)

        # Content frame for placing widgets inside the overlay
        self.content_frame = tk.Frame(self, bg=bg_color) # Inherits alpha
        self.content_frame.pack(fill=tk.BOTH, expand=True)

        # Create resize borders for all sides and corners
        border_width = 5

        # Create resize handles for all sides
        self.resize_n = tk.Frame(self, bg=bg_color, height=border_width, cursor="sb_v_double_arrow")
        self.resize_n.place(relx=0, rely=0, relwidth=1, height=border_width)

        self.resize_s = tk.Frame(self, bg=bg_color, height=border_width, cursor="sb_v_double_arrow")
        self.resize_s.place(relx=0, rely=1, relwidth=1, height=border_width, anchor="sw")

        self.resize_w = tk.Frame(self, bg=bg_color, width=border_width, cursor="sb_h_double_arrow")
        self.resize_w.place(relx=0, rely=0, width=border_width, relheight=1)

        self.resize_e = tk.Frame(self, bg=bg_color, width=border_width, cursor="sb_h_double_arrow")
        self.resize_e.place(relx=1, rely=0, width=border_width, relheight=1, anchor="ne")

        # Create resize handles for all corners
        self.resize_nw = tk.Frame(self, bg=bg_color, width=border_width, height=border_width, cursor="size_nw_se")
        self.resize_nw.place(relx=0, rely=0, width=border_width, height=border_width)

        self.resize_ne = tk.Frame(self, bg=bg_color, width=border_width, height=border_width, cursor="size_ne_sw")
        self.resize_ne.place(relx=1, rely=0, width=border_width, height=border_width, anchor="ne")

        self.resize_sw = tk.Frame(self, bg=bg_color, width=border_width, height=border_width, cursor="size_ne_sw")
        self.resize_sw.place(relx=0, rely=1, width=border_width, height=border_width, anchor="sw")

        self.resize_se = tk.Frame(self, bg=bg_color, width=border_width, height=border_width, cursor="size_nw_se")
        self.resize_se.place(relx=1, rely=1, width=border_width, height=border_width, anchor="se")

        # Bind mouse events for moving and resizing
        self.title_bar.bind("<ButtonPress-1>", self.start_move)
        self.title_bar.bind("<ButtonRelease-1>", self.stop_move)
        self.title_bar.bind("<B1-Motion>", self.do_move)

        # Bind resize events for all edges and corners
        self.resize_n.bind("<ButtonPress-1>", lambda e: self.start_resize_edge(e, "n"))
        self.resize_s.bind("<ButtonPress-1>", lambda e: self.start_resize_edge(e, "s"))
        self.resize_w.bind("<ButtonPress-1>", lambda e: self.start_resize_edge(e, "w"))
        self.resize_e.bind("<ButtonPress-1>", lambda e: self.start_resize_edge(e, "e"))
        self.resize_nw.bind("<ButtonPress-1>", lambda e: self.start_resize_edge(e, "nw"))
        self.resize_ne.bind("<ButtonPress-1>", lambda e: self.start_resize_edge(e, "ne"))
        self.resize_sw.bind("<ButtonPress-1>", lambda e: self.start_resize_edge(e, "sw"))
        self.resize_se.bind("<ButtonPress-1>", lambda e: self.start_resize_edge(e, "se"))

        for widget in [self.resize_n, self.resize_s, self.resize_w, self.resize_e,
                      self.resize_nw, self.resize_ne, self.resize_sw, self.resize_se]:
            widget.bind("<ButtonRelease-1>", self.stop_resize_edge)
            widget.bind("<B1-Motion>", self.do_resize_edge)

        # Variables to store drag start position and window size
        self._drag_start_x = 0
        self._drag_start_y = 0
        self._resize_start_x = 0
        self._resize_start_y = 0
        self._resize_start_width = 0
        self._resize_start_height = 0
        self._resize_direction = None
        self._resize_start_geometry = None

        self.is_visible_during_ocr = True # Flag (though not actively used in capture logic now)

    def update_color(self, new_color):
        """Update the color of all overlay components without changing window opacity."""
        if self.winfo_exists():
            try:
                # Update all components
                self.configure(bg=new_color)
                self.title_bar.configure(bg=new_color)
                self.content_frame.configure(bg=new_color)
                self.close_button.configure(bg=new_color)

                # Update all resize handles
                for widget in [self.resize_n, self.resize_s, self.resize_w, self.resize_e,
                               self.resize_nw, self.resize_ne, self.resize_sw, self.resize_se]:
                    widget.configure(bg=new_color)

                self.update_idletasks()
            except tk.TclError:
                pass  # Ignore errors if window is being destroyed
    def start_move(self, event):
        """Record starting position for window move."""
        self._drag_start_x = event.x
        self._drag_start_y = event.y

    def stop_move(self, event):
        """Reset drag start position."""
        self._drag_start_x = 0
        self._drag_start_y = 0

    def do_move(self, event):
        """Move window based on mouse drag."""
        if self._drag_start_x is not None and self._drag_start_y is not None:
            deltax = event.x - self._drag_start_x
            deltay = event.y - self._drag_start_y
            x = self.winfo_x() + deltax
            y = self.winfo_y() + deltay
            self.geometry(f"+{x}+{y}")

    def start_resize_edge(self, event, direction):
        """Record starting position, size, and direction for window resize."""
        self._resize_start_x = event.x_root
        self._resize_start_y = event.y_root
        self._resize_start_width = self.winfo_width()
        self._resize_start_height = self.winfo_height()
        self._resize_direction = direction
        self._resize_start_geometry = (self.winfo_x(), self.winfo_y(),
                                     self.winfo_width(), self.winfo_height())

    def stop_resize_edge(self, event):
        """Reset resize state."""
        self._resize_direction = None
        self._resize_start_geometry = None

    def do_resize_edge(self, event):
        """Resize window based on mouse drag from the active edge or corner."""
        if self._resize_direction is None or self._resize_start_geometry is None:
            return

        delta_x = event.x_root - self._resize_start_x
        delta_y = event.y_root - self._resize_start_y
        x, y, width, height = self._resize_start_geometry

        # Minimum size constraints
        min_width = 100
        min_height = 50

        # Handle resize based on direction
        if "n" in self._resize_direction:  # North (top)
            new_y = y + delta_y
            new_height = height - delta_y
            if new_height >= min_height:
                y = new_y
                height = new_height

        if "s" in self._resize_direction:  # South (bottom)
            new_height = height + delta_y
            if new_height >= min_height:
                height = new_height

        if "w" in self._resize_direction:  # West (left)
            new_x = x + delta_x
            new_width = width - delta_x
            if new_width >= min_width:
                x = new_x
                width = new_width

        if "e" in self._resize_direction:  # East (right)
            new_width = width + delta_x
            if new_width >= min_width:
                width = new_width

        self.geometry(f"{width}x{height}+{x}+{y}")

    def get_geometry(self):
        """Return the window's geometry as [x1, y1, x2, y2]."""
        try:
            if not self.winfo_exists(): return None # Check if window exists
            x = self.winfo_x()
            y = self.winfo_y()
            width = self.winfo_width()
            height = self.winfo_height()
            return [x, y, x + width, y + height]
        except tk.TclError:
            log_debug("Error getting geometry: Window likely destroyed.")
            return None

    def hide(self):
        """Hide the window."""
        if self.winfo_exists():
            self.withdraw()
        self.is_visible_during_ocr = False

    def show(self):
        """Show the window and ensure it's topmost."""
        if self.winfo_exists():
            self.deiconify()
            self.attributes("-topmost", True)
        self.is_visible_during_ocr = True

    def toggle_visibility(self):
        """Toggle the window's visibility."""
        if self.winfo_exists():
            if self.winfo_viewable():
                self.hide()
            else:
                self.show()
