# Project DSOCR
#
# dsocr_ui.py
# Providing GUI Interface of DS OCR
# by dof-studio/Nathmath
# Open Source Under Apache 2.0 License
# Website: https://github.com/dof-studio/DSOCR

import os
import re
import sys
import json
import lzma
import zstandard
import ctypes
import base64
import torch
import random
import pickle
import pandas
import requests
import datetime
import zstandard
from io import BytesIO
from copy import deepcopy
from datetime import datetime
import hashlib
from functools import lru_cache
from PIL import Image, ImageGrab
from pathlib import Path as Path
from typing import Any, Tuple
from numpy import array as make_array

# Debug Env (False for Production Use)
DSOCR_DEBUG = False

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QTextEdit, QListWidget, QComboBox, QProgressBar,
    QFileDialog, QLineEdit, QMenuBar, QMenu, QMessageBox, QDialog, QFormLayout,
    QRadioButton, QGroupBox, QScrollArea, QGraphicsScene, QGraphicsPixmapItem,
    QListWidget, QListWidgetItem, QWidget, QApplication,
    QMenu, QInputDialog, QVBoxLayout, QLabel
)

from PyQt6.QtGui import QPixmap, QClipboard, QImage, QPalette, QColor, QBrush, QPainter, QWheelEvent, QMouseEvent, QTextCursor, QIcon, QFont, QDesktopServices
from PyQt6.QtWidgets import QGraphicsBlurEffect
from PyQt6.QtCore import QThread, Qt, QUrl, QTimer, QRectF, QPoint, QPointF
from PyQt6.QtCore import pyqtSignal, pyqtSlot

# Crypto API
import dsocr_crypto
from dsocr_crypto import import_rsa_key, rsa_public_decrypt_raw
from dsocr_crypto import blake2b_derive_key, aes_cbc_decrypt

# Inference API
if not DSOCR_DEBUG:
    import dsocr_runmodel
    from dsocr_runmodel import DeepSeekOCRModel
    
# Post Processing API
if not DSOCR_DEBUG:
    import dsocr_postprocess
    from dsocr_postprocess import draw_outlines, drop_positional
    
# Inference API
if not DSOCR_DEBUG:
    import dsocr_custom_infer
    from dsocr_custom_infer import supports_bnb_nf4_
    from dsocr_custom_infer import supports_bnb_8bit_
    from dsocr_custom_infer import clear_positional_
    from dsocr_custom_infer import dsocr_custom_infer_
    from dsocr_custom_infer import TextAccumulator
else:
    # A mimic function
    def supports_bnb_nf4_():
        return True
    def supports_bnb_8bit_():
        return True
    def clear_positional_(text):
        return text
    
# Default Kernel Version
DSOCR_UI_DEFAULT_KRLVER = "0"

# Default GUI Version
DSOCR_UI_DEFAULT_GUIVER = "0.0.1"

# Default Model Path
if not DSOCR_DEBUG:
    DSOCR_UI_DEFAULT_MODELPATH = '../model/DeepSeek-OCR/'

# Default Working Directory Path
DSOCR_UI_DEFAULT_WDPATH = "../wd/o/"

# Default Font Path
DSOCR_UI_DEFAULT_FTPATH = "../fonts/SourceHanSansHWSC-VF.ttf"

# Default Res Path
DSOCR_UI_DEFAULT_RSPATH = "../res/"

# Default ICO Path
DSOCR_UI_DEFAULT_ICPATH = "../res/icon.jpg"

# Default Background Path
DSOCR_UI_DEFAULT_BGPATH = "../res/bgx.png"
DSOCR_UI_DEFAULT_LGPATH = "../res/lgx.png"


# Dump Project Save
def save(obj: Any, filename: str, *, kompress: Any = None, protocol: int | None = None, **kwargs):
    """
    Save a Python object to a file using pickle.
    Directly save without wrapping.
    """
    # Uncompress
    if kompress is None:
        with open(filename, 'wb') as f:
            pickle.dump(obj, f, protocol = protocol)
    # Compress
    else:
        with kompress.open(filename, 'wb', **kwargs) as f:
            pickle.dump(obj, f, protocol = protocol)


# Dump Project Load
def load(filename: str, *, kompress: Any = None) -> Any:
    """
    Load a Python object from a pickle file.
    Generally loading. Try to unwrap if possible
    
    Exception:
        Throw a ValueError when in the dumping mode and failed to
        pass the hash test.
    """
    # Uncompress
    if kompress is None:
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
    else:
        with kompress.open(filename, 'rb') as f:
            obj = pickle.load(f)
    
    # No need to unwrap
    if isinstance(obj, dict) == False:
        return obj
    elif isinstance(obj, dict) == True and obj.get("~attr~", None) is None:
        return obj
    
    # Need to unwrap
    if isinstance(obj, dict) == True and obj.get("~attr~", None) == "~dump~":
        if obj.get("~hash~", None) is None:
            raise ValueError("Corrupted dumpped file. Hash attribute has Nonetype.")
        elif isinstance(obj.get("~hash~", None), str) == False:
            raise ValueError("Corrupted dumpped file. Hash attribute has Non-string type.")
        if obj.get("data", None) is None:
            raise ValueError("Corrupted dumpped file. Data attribute is Nonetype.")
        if obj.get("~hash~", None) != str(hash(obj.get("data"))):
            raise ValueError("Corrupted dumpped file. Data hash mismatched.")
        return obj["data"] 
    
    else:
        return obj


# Main Executable Button
class ExecutableButton(QPushButton):
    
    def __init__(self, text, parent=None):
        super().__init__(text, parent)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self.on_right_click() # bind: on_right_click
        else:
            super().mousePressEvent(event)

    def on_right_click(self):
        pass # Override outside


# Image Label
class QImageLabel(QLabel):
    
    def __init__(self, parent=None, welcome=r"单击上载图片/长按从剪贴板里复制", *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.welcome = welcome
        self.setText(welcome)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._pixmap: QPixmap | None = None
        self._scale: float = 1.0
        
        # Offset of the image's top-left corner relative to the label's client area
        self._offset: QPointF = QPointF(0.0, 0.0)
        self._dragging: bool = False
        self._last_pos: QPointF = QPointF(0.0, 0.0)
        # Used to determine if it's a click rather than a drag
        self._press_pos: QPointF = QPointF(0.0, 0.0)
        # Counts as clicks within a pixel
        self.click_threshold: float = 5.0
        # Externally bindable callback function, signature can be func()
        self.on_click = None
        # Externally bindable callback function, signature can be func()
        self.on_long_press = None
        # If done long press, then invalidate click event
        self._long_press_affected = False
        # Timer for long press detection
        self._long_press_timer = QTimer()
        self._long_press_timer.setSingleShot(True)
        self._long_press_timer.timeout.connect(self._handle_long_press)
        self._long_press_duration = 1000  # ms
        
        # Whether the user has manually zoomed/dragged from original position
        self._user_interacted: bool = False  
        
        # Whether the user has dragged after clicking
        self._is_dragged_after_click: bool = False
        
        # Default translation bounds enable/disable, set to True to limit excessive whitespace
        self._constrain_to_bounds = True

    def clearImage(self):
        """
        Remove the current image and restore the text display to the initial state.
        """
        self._pixmap = None
        self._scale = 1.0
        self._offset = QPointF(0.0, 0.0)
        self._dragging = False
        self._last_pos = QPointF(0.0, 0.0)
        self._user_interacted = False
    
        self.setText(self.welcome)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
    
        self.update()

    def reset_view(self):
        """
        Force refit and reset the interactive state to non-interactive (equivalent to re-calling the fit part of setImage)
        """
        if not self._pixmap:
            return
        vw = max(1, self.width())
        vh = max(1, self.height())
        pw = self._pixmap.width()
        ph = self._pixmap.height()
        self._scale = min(vw / pw, vh / ph)
        scaled_w = pw * self._scale
        scaled_h = ph * self._scale
        self._offset = QPointF((vw - scaled_w) / 2.0, (vh - scaled_h) / 2.0)
        self._user_interacted = False
        self.update()

    def setImage(self, pixmap: QPixmap):
        self._pixmap = pixmap
        if not pixmap or pixmap.isNull():
            self._scale = 1.0
            self._offset = QPointF(0.0, 0.0)
            self.update()
            return
    
        vw = max(1, self.width())
        vh = max(1, self.height())
        pw = pixmap.width()
        ph = pixmap.height()
    
        # fit / contain: Ensure the entire image is visible and enlarged as much as possible
        self._scale = min(vw / pw, vh / ph)
        scaled_w = pw * self._scale
        scaled_h = ph * self._scale
        self._offset = QPointF((vw - scaled_w) / 2.0, (vh - scaled_h) / 2.0)
    
        self._dragging = False
        self._last_pos = QPointF(0.0, 0.0)
        self._user_interacted = False
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().color(self.backgroundRole()))
        if not self._pixmap or self._pixmap.isNull():
            super().paintEvent(event)
            return

        # Calculate the drawing area and scaled size
        w = self._pixmap.width() * self._scale
        h = self._pixmap.height() * self._scale
        target = QRectF(self._offset.x(), self._offset.y(), w, h)
        painter.drawPixmap(target, self._pixmap, QRectF(0, 0, self._pixmap.width(), self._pixmap.height()))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._pixmap:
            return
    
        if not self._user_interacted:
            vw = max(1, self.width())
            vh = max(1, self.height())
            pw = self._pixmap.width()
            ph = self._pixmap.height()
            # Refit the image and center it (same rules as setImage)
            self._scale = min(vw / pw, vh / ph)
            scaled_w = pw * self._scale
            scaled_h = ph * self._scale
            self._offset = QPointF((vw - scaled_w) / 2.0, (vh - scaled_h) / 2.0)
            self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._last_pos = event.position()
            self._press_pos = event.position()
            self._long_press_timer.start(self._long_press_duration)
            self._long_press_affected = False
            self._is_dragged_after_click = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._dragging and self._pixmap:
            self._user_interacted = True
            self._is_dragged_after_click = True
            pos = event.position()
            delta = pos - self._last_pos
            self._last_pos = pos
            self._offset += delta
            if self._constrain_to_bounds:
                self._clamp_offset()
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self._long_press_timer.stop()
            
            # If not a long press, then try checking click event
            if self._long_press_affected == False:
                # Determine whether it is dragged
                release_pos = event.position()
                dist = (release_pos - self._press_pos).manhattanLength()
                if dist <= self.click_threshold and callable(self.on_click):
                    try:
                        self.on_click()
                    except Exception:
                        pass
            super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        if not self._pixmap:
            return
        self._user_interacted = True
        # Mouse coordinates (relative to the widget)
        cursor = event.position()
        # Use the cursor as the zoom center: calculate the cursor's position in the image coordinate system (pixel position before zooming)
        old_scale = self._scale
        # Zoom up/ down
        num_degrees = event.angleDelta().y() / 8
        num_steps = num_degrees / 15
        factor = 1.15 ** num_steps
        self._scale *= factor
        # Limit zoom range
        self._scale = max(0.05, min(20.0, self._scale))

        # Make the image point pointed by the cursor consistent before and after zooming, and update the offset
        if old_scale != 0:
            img_point_x = (cursor.x() - self._offset.x()) / old_scale
            img_point_y = (cursor.y() - self._offset.y()) / old_scale
            self._offset = QPointF(cursor.x() - img_point_x * self._scale,
                                   cursor.y() - img_point_y * self._scale)

        if self._constrain_to_bounds:
            self._clamp_offset()
        self.update()
        super().wheelEvent(event)

    def _clamp_offset(self):
        """
        Limits the offset value to prevent the image from completely moving outside the viewport after zooming or panning.
        Allows some white space, but tries to bring the image edges back into view.
        """
        if not self._pixmap:
            return
        w = self._pixmap.width() * self._scale
        h = self._pixmap.height() * self._scale
        vw = self.width()
        vh = self.height()
        # min/max allowed offset.x such that image still covers the widget or is centered
        min_x = min(0.0, vw - w)
        max_x = max(0.0, vw - w) if w < vw else 0.0
        min_y = min(0.0, vh - h)
        max_y = max(0.0, vh - h) if h < vh else 0.0

        # If image is larger than viewport, allow panning across whole image
        if w > vw:
            min_x = vw - w
            max_x = 0.0
        if h > vh:
            min_y = vh - h
            max_y = 0.0

        # Clamp
        ox = self._offset.x()
        oy = self._offset.y()
        ox = max(min_x, min(max_x, ox))
        oy = max(min_y, min(max_y, oy))
        self._offset = QPointF(ox, oy)

    def _handle_long_press(self) -> bool:
        
        # If dragged or scrolled, no
        if self._is_dragged_after_click == True:
            return
        
        # If no other interface, regard as a long press
        self._long_press_affected = True
        # Already triggered press event
        if callable(self.on_long_press):
            try:
                self.on_long_press()
            except Exception:
                pass

    
    def get_scale(self) -> float:
        return self._scale

    def get_offset(self) -> QPointF:
        return self._offset


# Select Task State Popup
class TaskStatePopup(QListWidget):
    """
    Small popup list that behaves like a combo box popup.
    It waits for mouse release to confirm selection. If release is outside any item, it cancels.
    """
    selection_made = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, options, parent=None):
        super().__init__(parent)

        # Show as popup windows
        self.setWindowFlags(Qt.WindowType.Popup)

        # Items of states
        self.addItems(options)
        
        # Track the mouse, when mouse is on then focus
        self.setMouseTracking(True)
        self.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setSpacing(2)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)

        # Shrink size
        self.adjust_size_to_contents()

    def adjust_size_to_contents(self):
        # Adjust the popup size tightly around its items.
        self.setUniformItemSizes(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.resize(self.sizeHintForColumn(0) + 40, self.sizeHintForRow(0) * self.count() + 20)

    def mouseMoveEvent(self, e):
        # If on item, then set the item as selected
        # If out of item, then de-select 
        item = self.itemAt(e.position().toPoint())
        if item:
            self.setCurrentItem(item)
        else:
            self.setCurrentItem(None)
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        # When mouse releases, make a decision based on the selected one 
        # or cancelled if no one is selected
        pos = e.position().toPoint()
        item = self.itemAt(pos)
        if item:
            self.selection_made.emit(item.text())
        else:
            self.cancelled.emit()
        self.hide()
        super().mouseReleaseEvent(e)

    def leaveEvent(self, e):
        # Called when the mouse leaves the popup area
        self.cancelled.emit()
        self.hide()
        e.accept()
        

# Main GUI Task History Tracker
class TaskHistoryList(QListWidget):
    """
    QListWidget subclass has:
      - double-click on empty area -> emits new_task_requested and calls self._new_task() if present
      - double-click on item -> emits enter_task(item) and calls self._enter_task(item) if present
      - long-press (>1000 ms) on an item -> shows popup state chooser, drag & release to select/cancel
      - right-click on item -> context menu (rename, delete)
      - colors items according to their state
    """

    _new_task_requested_em = pyqtSignal() # Binding
    _enter_task_em = pyqtSignal(QListWidgetItem) # Binding
    _state_changed_em = pyqtSignal(QListWidgetItem, int) # Binding
    _rename_requested_em = pyqtSignal(QListWidgetItem, str) # Binding
    _delete_requested_em = pyqtSignal(QListWidgetItem) # Binding
    _delete_all_elems_em = pyqtSignal() # Binding
    
    # Three states:
    # > reset states (not ready): no image is provided (-1)
    # > ready states (ready but unfinished): image is there but not executed (0)
    # > comleted status (finished): image is processed and generated (1)
    DEFAULT_STATES = ["reset", "ready", "completed"]
    DEFAULT_STATES_CNMAPING = ["初始化", "未处理", "已处理"]
    STATE_COLORS = {
        "reset": QColor("black"),      # white when dark mode
        "ready": QColor("red"),        # as is
        "completed": QColor("green"),  # as is
    }
    LONG_PRESS_MS = 600 # 600 ms for long press

    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Long-press timer
        self._long_press_timer = QTimer(self)
        self._long_press_timer.setSingleShot(True)
        self._long_press_timer.timeout.connect(self._on_long_press_timeout)
        
        # Mouse press detection
        self._pressed_item = None
        self._pressed_pos = None
        self._popup = None
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_custom_context_menu)
        self.itemDoubleClicked.connect(self._on_item_double_clicked)
        
    # Reset Colors (after applying theme change)
    def _reset_theme_colorset(self, mode="dark"):
        if mode == "dark":
            self.STATE_COLORS = {
                    "reset": QColor("white"),      # white when dark mode
                    "ready": QColor("red"),        # as is
                    "completed": QColor("green"),  # as is
                }
        elif mode == "light":
            self.STATE_COLORS = {
                    "reset": QColor("black"),      # white when dark mode
                    "ready": QColor("red"),        # as is
                    "completed": QColor("green"),  # as is
                }
        return
    
    # Get default color under current theme
    def _get_default_color(self):
        return self.STATE_COLORS["reset"]
    
    # Helper, to convert mapping to original states
    def _state_mapping_cntoen(self, state:str="初始化"):
        if isinstance(state, str) == False:
            raise TypeError("Only accepts an str state in cntoen mode")
        if state == "初始化":
            return "reset"
        elif state == "未处理":
            return "ready"
        elif state == "已处理":
            return "completed"
        else:
            # By default: reset 
            return "reset"
        
    # State Mapping INT to STR
    def _state_mapping_tostr(self, state: int = -1):
        if isinstance(state, int) == False:
            raise TypeError("Only accepts an int state in tostr mode")
        if state == -1:
            return "reset"
        elif state == 0:
            return "ready"
        elif state == 1:
            return "completed"
        else:
            # By default: reset
            return "reset"
        
    # State Mapping STR to INT
    def _state_mapping_toint(self, state:str="reset"):
        if isinstance(state, str) == False:
            raise TypeError("Only accepts an str state in toint mode")
        if state == "reset":
            return -1
        elif state == "ready":
            return 0
        elif state == "completed":
            return 1
        else:
            # By default: reset (-1)
            return -1

    # Add a task
    def add_task(self, text, state:str|int="reset"):
        if isinstance(state, int):
            state = self._state_mapping_tostr(state)
        curitem = self.currentItem()
        # Create a new item
        item = QListWidgetItem(text)
        item.setData(Qt.ItemDataRole.UserRole, state)
        self._apply_color(item)
        # Still stays at current
        self.setCurrentIndex(self.indexFromItem(curitem))
        super().addItem(item)
        return item
    
    # Add a task imitator
    def addItem(self, text, *, state:str|int="reset"):
        return self.add_task(text, state = state)

    # Manually set task state WITHOUT emitting signal
    def set_task_state_naive_noemit_(self, item, state:str|int="reset"):
        if isinstance(state, int):
            state = self._state_mapping_tostr(state)
        item.setData(Qt.ItemDataRole.UserRole, state)
        self._apply_color(item)
        
    # Manually set task state with emitting signal
    def set_task_state(self, item, state:str|int="reset"):
        if isinstance(state, int):
            state = self._state_mapping_tostr(state)
        item.setData(Qt.ItemDataRole.UserRole, state)
        self._apply_color(item)
        
        # Emit signal (using int adjustment)
        self._state_changed_em.emit(item, self._state_mapping_toint(state))
        
    # Internal, apply color scheme
    def _apply_color(self, item):
        state = item.data(Qt.ItemDataRole.UserRole) or "reset"
        color = self.STATE_COLORS.get(state, self._get_default_color())
        item.setForeground(QBrush(color))
        
    # Internal, get item state int mode
    def _get_item_state_int(self, item):
        state = item.data(Qt.ItemDataRole.UserRole) or "reset"
        return self._state_mapping_toint(state)
    
    # Try to remove (take) an item (This should be called by Parent)
    def _try_take_item(self, item):
        return self.takeItem(self.row(item))
        
    # Reset theme (API, called when GUI theme is changed)
    def reset_theme(self, theme="dark"):
        self._reset_theme_colorset(mode=theme)
        for i in range(self.count()):
            item = self.item(i)
            self._apply_color(item)

    # Mouse / long-press logic
    def mousePressEvent(self, e):
        pos = e.position().toPoint()
        self._pressed_pos = pos
        self._pressed_item = self.itemAt(pos)
        if e.button() == Qt.MouseButton.LeftButton and self._pressed_item is not None:
            # start long-press detection
            self._long_press_timer.start(self.LONG_PRESS_MS)
        super().mousePressEvent(e)

    # Mouse / long-press logic
    def mouseMoveEvent(self, e):
        # Detect if a long-press (not moving away)
        if self._pressed_pos is not None:
            dist = (e.position().toPoint() - self._pressed_pos).manhattanLength()
            if dist > QApplication.startDragDistance():
                # moved too far, cancel long-press
                if self._long_press_timer.isActive():
                    self._long_press_timer.stop()
        super().mouseMoveEvent(e)

    # Mouse / long-press logic
    def mouseReleaseEvent(self, e):
        # Detect if a long-press
        if self._long_press_timer.isActive():
            # not a long press (released earlier) -> cancel timer
            self._long_press_timer.stop()
        # if popup is shown, let popup handle selection on its own mouseReleaseEvent
        super().mouseReleaseEvent(e)
        self._pressed_item = None
        self._pressed_pos = None
        
    # Double click on empty: new task
    def mouseDoubleClickEvent(self, e):
        pos = e.position().toPoint()
        item = self.itemAt(pos)
        if item is None:
            # double-click on empty area: new task
            self._new_task_requested_em.emit()
            # also call method to new task
            fn = getattr(self, "_new_task", None)
            if callable(fn):
                fn()
        else:
            # let itemDoubleClicked signal handle it (connected)
            super().mouseDoubleClickEvent(e)
            
    # Double click on item: select and change task
    def _on_item_double_clicked(self, item):
        fn = getattr(self, "_enter_task", None)
        if callable(fn):
            fn(item)
        # First call underlying then call enter task
        # Since we rely on the correct currentItem
        self._enter_task_em.emit(item)

    # Long press timeout show popup to select the state
    def _on_long_press_timeout(self):
        # called when press lasts LONG_PRESS_MS; show popup to select state
        if not self._pressed_item:
            return
        item = self._pressed_item
        rect = self.visualItemRect(item)
        global_pos = self.viewport().mapToGlobal(rect.center())
        self._show_state_popup(global_pos, item)

    # Show popup window for user to select state (deprecated now for unknown bugs）
    def _show_state_popup(self, global_pos: QPoint, item: QListWidgetItem):
        if self._popup and self._popup.isVisible():
            self._popup.hide()

        # Item having a current image (deduction)
        item_has_current_image = item.data(Qt.ItemDataRole.UserRole) != "reset"
        options = []
        if item_has_current_image == False:
            # Only can be reset mode
            options.append("初始化") # reset, 0
        else:
            # Add all
            for s in self.DEFAULT_STATES_CNMAPING:
                options.append(s)
        self._popup = TaskStatePopup(options, parent=self)

        # place popup just right to the cursor
        self._popup.move(self.cursor().pos())
        
        # connect signals
        self._popup.selection_made.connect(lambda state: self._on_state_popup_selected(item, state))
        self._popup.cancelled.connect(lambda: None)
        self._popup.show()

    # Long press timeout and state selected
    def _on_state_popup_selected(self, item, state):
        # apply state change
        self.set_task_state(item, self._state_mapping_cntoen(state))

    # Right Click Context menu
    def _on_custom_context_menu(self, pos):
        item = self.itemAt(pos)
        if item is None:
            return
        
        # Item having a current image (deduction)
        item_has_current_image = item.data(Qt.ItemDataRole.UserRole) != "reset"
        
        # Right click menu
        menu = QMenu(self)
        rename_action = menu.addAction("重命名")
        delete_action = menu.addAction("删除")
        deleta_all_action = menu.addAction("删除全部")
        change_state_menu = menu.addMenu("重设状态")
        # Insert Chinese names of states
        # But note that we don't usually show every option
        if item_has_current_image == False:
            # Only can be reset mode
            change_state_menu.addAction("初始化") # reset, 0
        else:
            # Add all
            for s in self.DEFAULT_STATES_CNMAPING:
                change_state_menu.addAction(s)

        action = menu.exec(self.viewport().mapToGlobal(pos))
        if action is None:
            return
        text = action.text()
        
        # rename logic
        if text == "重命名":
            current = item.text()
            new_text, ok = QInputDialog.getText(self, "任务重命名", "新的名字:", text=current)
            if ok and new_text:
                self._rename_requested_em.emit(item, new_text.strip())
                # Must set new name after this! I will use old name to locate items
                # The name of the item will be set with in the function
        
        # delete logic
        elif text == "删除":
            row = self.row(item)
            
            # Same logic, first delete the record
            if self.count() > 1:
                pass # This logic is implemented in _try_take_elem(item)
                     # And will called by emit function
            else:
                QMessageBox.warning(None, "错误", "你必须至少保留一个任务.")
                pass # We must at least keep 1
                     # Same rule followed by frontend

            # The emit function will follow the checking: count > 1
            self._delete_requested_em.emit(item)
        
        # state adjustment logic
        elif text in self.DEFAULT_STATES_CNMAPING:
            
            # Covert to English and set state
            self.set_task_state(item, self._state_mapping_cntoen(text))
            # This set_task_state will change the appearance here
            # But you have to manually adjust flag in main class representation 

        # delete all logic
        elif text == "删除全部":

            # The emit function will trigger and the elements here will be removed from outside
            self._delete_all_elems_em.emit()

        else:
            # Unknowns
            pass


# Setting GUI
class Theme_SettingsDialog(QDialog):
    
    def __init__(self, parent=None):

        super().__init__(parent)
        self.setWindowTitle("界面主题设置")
        self.setModal(True)
        self.setFixedSize(300, 200)
        
        layout = QFormLayout(self)

        # Theme selection
        theme_group = QGroupBox("界面主题")
        theme_layout = QVBoxLayout(theme_group)
        
        self.light_radio = QRadioButton("澄空 (Light)")
        self.dark_radio = QRadioButton("静海 (Dark)")
        self.who_is_baka = "NathMath Desu"
        theme_layout.addWidget(self.light_radio)
        theme_layout.addWidget(self.dark_radio)
        
        layout.addRow(theme_group)
        
        # Apply theme button
        apply_btn = QPushButton("确认更改")
        apply_btn.clicked.connect(self.apply_theme_settings)
        layout.addRow(apply_btn)

        # Default to current
        if self.parent().current_theme == 'light':
            self.light_radio.setChecked(True)
        else:
            self.dark_radio.setChecked(True)
        
    def update_theme_preview(self):
        pass  # Placeholder for preview
    
    def apply_theme_settings(self):
        theme = "dark" if self.dark_radio.isChecked() else "light"
        self.parent().set_theme(theme.strip())
        self.accept()


# InferEnv Setting GUI
class InferEnv_SettingsDialog(QDialog):
    
    def __init__(self, parent=None):

        super().__init__(parent)
        self.setWindowTitle("推理环境设置")
        self.setModal(True)
        self.setFixedSize(300, 250)
        
        layout = QFormLayout(self)

        # Inference Env Setting
        inferenv_group = QGroupBox("推理环境")
        inferenv_layout = QVBoxLayout(inferenv_group)
            
        self.inferenv_cuda_nf4 = QRadioButton("CUDA (nf4)")
        self.inferenv_cuda_int8 = QRadioButton("CUDA (int8)")
        self.inferenv_cuda_bf16 = QRadioButton("CUDA (bf16)")
        self.inferenv_cuda_fp16 = QRadioButton("CUDA (fp16)")
        self.inferenv_cpu = QRadioButton("CPU (float32)")
        self.inferenv_bestavil = QRadioButton("自动推导最优选项")
        inferenv_layout.addWidget(self.inferenv_cuda_nf4)
        inferenv_layout.addWidget(self.inferenv_cuda_int8)
        inferenv_layout.addWidget(self.inferenv_cuda_bf16)
        inferenv_layout.addWidget(self.inferenv_cuda_fp16)
        inferenv_layout.addWidget(self.inferenv_cpu)
        inferenv_layout.addWidget(self.inferenv_bestavil)
        if self.parent()._inference_device_env().type.startswith("cuda"):
            # nf4
            if DSOCR_DEBUG or supports_bnb_nf4_():
                self.inferenv_cuda_nf4.setCheckable(True)
            else:
                self.inferenv_cuda_nf4.setCheckable(False)
            # int8
            if DSOCR_DEBUG or supports_bnb_8bit_():
                self.inferenv_cuda_int8.setCheckable(True)
            else:
                self.inferenv_cuda_int8.setCheckable(False)
            # bf16 support
            if self.parent()._inference_dtype_bf16_support() == True:
                self.inferenv_cuda_bf16.setCheckable(True)
            else:
                self.inferenv_cuda_bf16.setCheckable(False)
            # assume at least support fp16
            self.inferenv_cuda_fp16.setCheckable(True)
        else:
            self.inferenv_cuda_nf4.setCheckable(False)
            self.inferenv_cuda_int8.setCheckable(False)
            self.inferenv_cuda_bf16.setCheckable(False)
            self.inferenv_cuda_fp16.setCheckable(False)
            
        layout.addRow(inferenv_group)    
        
        # Apply theme button
        apply_btn = QPushButton("确认切换")
        apply_btn.clicked.connect(self.apply_inferenv_settings)
        layout.addRow(apply_btn)

        # Default setting
        if self.parent().inference_device.type.lower() == "cpu":
            self.inferenv_cpu.setChecked(True)
        else:
            # GPU, let's check quant
            if self.parent().inference_quant == "nf4":
                self.inferenv_cuda_nf4.setChecked(True) 
            elif self.parent().inference_quant == "bnb_8bit":
                self.inferenv_cuda_int8.setChecked(True)
            elif self.parent().inference_quant == "bfloat16":
                self.inferenv_cuda_bf16.setChecked(True)
            elif self.parent().inference_quant == "float16":
                self.inferenv_cuda_fp16.setCheckable(True)
            else:
                self.inferenv_cpu.setChecked(True)        

    def update_inferenv_preview(self):
        pass  # Placeholder for preview

    def apply_inferenv_settings(self):

        # If model is busy, sorry we can't do it
        if self.parent().model_isbusy:
            QMessageBox.warning(None, "错误", "模型正忙,请稍后切换推理环境")
            self.reject()
        
        device = None
        quant = None

        # Check nf4
        if self.inferenv_cuda_nf4.isChecked():
            device = "cuda"
            quant = "nf4"
        elif self.inferenv_cuda_int8.isChecked():
            device = "cuda"
            quant = "bnb_8bit"
        elif self.inferenv_cuda_bf16.isChecked():
            device = "cuda"
            quant = "bfloat16"
        elif self.inferenv_cuda_fp16.isChecked():
            device = "cuda"
            quant = "float16"
        elif self.inferenv_cpu.isChecked():
            device = "cpu"
            quant = None
        elif self.inferenv_bestavil.isChecked():
            pass # two Nones
        else:
            # Impossible, but regard as CPU
            device = "cpu"
            quant = None

        # Set env
        if device is not None:
            self.parent().update_infenv(torch.device(device), quant)
        else:
            # Auto deduction
            self.parent().update_infenv(None, None)
        self.accept()


# Async Task Executor
class AsyncTaskExecutor(QThread):
    """
    Asynchronous task executor thread for handling API calls.
    Emits signals for progress updates and log appends.
    """
    progress_updated = pyqtSignal(int)
    log_appended = pyqtSignal(str)
    task_completed = pyqtSignal(bool)  # bool: success

    def __init__(self, api_call_func):
        super().__init__()
        self.api_call_func = api_call_func  # User-provided function to insert logic

    def run(self):
        try:
            # Emit progress 75%
            for i in range(76):
                self.msleep(5)
                self.progress_updated.emit(i)
                
            self.api_call_func()
            
            # Complete the progress bar
            for i in range(76, 101, 1):
                self.msleep(1)
                self.progress_updated.emit(i)
            
            # Simulate async execution; replace with actual API call
            # User inserts logic here: self.api_call_func(resolution_type, task_type, image_path)
            # for i in range(101):
            #    self.msleep(50)  # Simulate progress
            #    self.progress_updated.emit(i)
            self.task_completed.emit(True)
            
        except Exception as e:
            # Log error here
            self.log_appended.emit(f"{'='*25}\nERROR:\nFailed because that {str(e)}\n\nIf you see this Error, please give feedback to our developer!")
            self.task_completed.emit(False)


# Main Image Processing GUI
class ImageProcessingGUI(QMainWindow):
    
    def __init__(self, 
                 *, 
                 wd: str = None, 
                 theme: str = None,
                 inference_device: str = None, 
                 quant: str = None,
                 enable_magic: str = None,
                 **kwargs):
        """
        Create the main Form of DS OCR GUI.
        
        Parameter:
            wd: str, default working directory, if not set, use DSOCR_UI_DEFAULT_WDPATH
            theme: str, theme description of the main form, if not set, use "dark" by default
            inference_device: str, the device used to do inference, if not set, then deduct the best avail
            quant: str, quantization mode, for example, "nf4" or "bnb_8bit", if not set, then deduct the best avail
            enable_magic: str, if display and show magic functionality. Can be None for no. "all" for all, "gal" for gal only
        """
        
        # Inference device can be set or left as None for auto detection
        
        super().__init__()
        # By Nathmath @ dof-studio
        self.setWindowTitle("DS OCR 图片转文字工具 (初始化中...)")
        self.setGeometry(100, 100, 900, 600)
        # Initialize size 900 * 600
        
        ###############################################################
        #
        # Path Related
        
        # Current Working Directory (by Default, WD Path)
        self.current_wd = wd
        if self.current_wd is None:
            # Falls back to default
            self.current_wd = DSOCR_UI_DEFAULT_WDPATH
            if os.path.exists(self.current_wd) == False:
                os.makedirs(self.current_wd, exist_ok=True)
        if os.path.exists(self.current_wd) == False:
            # Try create it, if failed, use default
            try:
                os.makedirs(self.current_wd, exist_ok=False)
            except:
                self.current_wd = DSOCR_UI_DEFAULT_WDPATH
                if os.path.exists(self.current_wd) == False:
                    os.makedirs(self.current_wd, exist_ok=True)
                    
        # Create /tmp if needed
        if os.path.exists(os.path.join(self.current_wd, "tmp")) == False:
            try:
                os.makedirs(os.path.join(self.current_wd, "tmp"), exist_ok=True)
            except:
                pass
                    
        # Placeholder of task imaging folder
        self.last_task_img_folder = None # if None, will be by default open at self.current_wd
        # by default, to wd, and will set it when saving an img

        # Structure:
        #   wd/
        #   wd/tmp/
        #   wd/o/<hash task>/...     
        #        

        ###############################################################
        #     
        # Image Display
                
        # Current Image and Last Image
        self.current_image_path = None  # [ts]
        self.last_image_path = None     # [ts] for one step back

        ###############################################################
        #
        # Main Form

        # Central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Background image (Placeholder, will replaced by theme)
        self.set_background_image(DSOCR_UI_DEFAULT_BGPATH)
        # Must be after initializng central_widget
        
        # Main Content Area
        self.content_layout = QHBoxLayout()
        
        # Left: Upload Image
        self.left_widget = QWidget()
        self.left_widget.setMinimumWidth(400)
        self.left_layout = QVBoxLayout(self.left_widget)
        self.left_layout.addWidget(QLabel("上传图片"))
        
        # Left: Image Label
        self.image_label = QImageLabel()
        # Bind: click to select a new image
        self.image_label.on_click = self.upload_image
        # Bind: long press to copy image from clipboard
        self.image_label.on_long_press = self.load_image_from_clipboard
        self.image_label.setMinimumSize(300, 500)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_layout.addWidget(self.image_label, 1) # Allow stretch
        
        # Create a horizontal layout for the buttons
        self.button_layout = QHBoxLayout()
        
        # Left: Copy the Image Button
        self.copy_image_btn = QPushButton("复制图片")
        self.copy_image_btn.clicked.connect(self.copy_image_to_clipboard)
        self.copy_image_btn.setEnabled(False)
        self.button_layout.addWidget(self.copy_image_btn)
        
        # Left: Back to Last Image Button
        self.rollback_image_btn = QPushButton("回滚图片")
        self.rollback_image_btn.clicked.connect(self.rollback_last_image)
        self.rollback_image_btn.setEnabled(False)
        self.button_layout.addWidget(self.rollback_image_btn)
        
        # Left: Clear Image Button
        self.clr_image_btn = QPushButton("清除图片")
        self.clr_image_btn.clicked.connect(self.clear_image)
        self.clr_image_btn.setEnabled(False)
        self.button_layout.addWidget(self.clr_image_btn)
        
        # Add the horizontal layout to your existing left_layout
        self.left_layout.addLayout(self.button_layout)
        
        self.left_layout.addStretch()
        self.content_layout.addWidget(self.left_widget, 1)
        
        # Center: Output Task - Expanded height
        self.center_widget = QWidget()
        self.center_widget.setMinimumWidth(400)
        self.center_layout = QVBoxLayout(self.center_widget)
        self.center_layout.addWidget(QLabel("文本输出"))
        self.current_output_text = QTextEdit() # [ts]
        self.current_output_text.setReadOnly(True) # By Nathmath @ dof-studio
        # Removed maximumHeight to allow expansion
        self.center_layout.addWidget(self.current_output_text, 1)  # Stretch factor 1
        
        # Create a horizontal layout for the buttons
        self.button_layout_c = QHBoxLayout()
        
        # Center: Copy Original Output Text Button
        self.copy_output_btn = QPushButton("复制原始输出")
        self.copy_output_btn.clicked.connect(self.copy_output_to_clipboard)
        self.button_layout_c.addWidget(self.copy_output_btn)
        
        # Center: Copy the Clear OCR Output Button
        self.copy_c_output_btn = QPushButton("复制无位置输出")
        self.copy_c_output_btn.clicked.connect(self.copy_cleared_output_to_clipboard)
        self.button_layout_c.addWidget(self.copy_c_output_btn)
        
        # Center: Copy the Clear OCR Output Button
        self.clr_output_btn = QPushButton("清空输出")
        self.clr_output_btn.clicked.connect(self.clear_log)
        self.button_layout_c.addWidget(self.clr_output_btn)
        
        # Add the horizontal layout to your existing center_layout
        self.center_layout.addLayout(self.button_layout_c)
        
        self.center_layout.addStretch()
        self.content_layout.addWidget(self.center_widget, 1)
        
        # Right: Task Management & Control
        self.right_widget = QWidget()
        self.right_widget.setFixedWidth(350)
        self.right_layout = QVBoxLayout(self.right_widget)
        
        # Right: History / NEW
        self.history_layout = QHBoxLayout()
        self.history_layout.addWidget(QLabel("任务列表"))
        self.right_layout.addLayout(self.history_layout)
        
        # Right: History Widget
        self.history_list = TaskHistoryList()
        self.history_list._new_task_requested_em.connect(self.create_new_task)
        self.history_list._enter_task_em.connect(self.switch_to_task)
        self.history_list._state_changed_em.connect(self.change_certain_task_state)
        self.history_list._rename_requested_em.connect(self.rename_certain_task)
        self.history_list._delete_requested_em.connect(lambda item: self.remove_certain_task(item, sendtolist=True)) # Must send back since it is not handled in the child
        self.history_list._delete_all_elems_em.connect(self.remove_all_tasks)
        self.right_layout.addWidget(self.history_list, 1)  # Stretch
        
        # Control Panel
        self.control_layout = QGridLayout()
        self.control_layout.addWidget(QLabel("控制面板"), 0, 0, 1, 2,)
        
        # Resolution Combo List (Selector)
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["默认", # redirect to Base
                                        "Default",
                                        "Tiny", 
                                        "Small", 
                                        "Base",
                                        "Large",
                                        "Super",
                                        "Ultra",
                                        "Experimental"])
        self.resolution_combo.currentIndexChanged.connect(self._control_panel_changed_base)
        self.control_layout.addWidget(QLabel("分辨率:"), 1, 0)
        self.control_layout.addWidget(self.resolution_combo, 1, 1)# v: [ts]
        
        # Task Name Combo List (Selector)
        self.task_type_combo = QComboBox()
        self.task_type_combo.addItems(["自由OCR",
                                       "图片转文字", 
                                       "图片转Markdown",
                                       "图表解读",
                                       "图片描述",
                                       "全图找物",
                                       "全图翻译",
                                       "全图问答",
                                       "自定义"])
        self.task_type_combo.currentIndexChanged.connect(self._control_panel_changed_base)
        self.control_layout.addWidget(QLabel("任务:"), 2, 0)
        self.control_layout.addWidget(self.task_type_combo, 2, 1) # v: [ts]
        
        # Customizable Prompt
        self.user_prompt_ipt = QLineEdit()
        self.user_prompt_ipt.setPlaceholderText("")
        self.user_prompt_ipt.setMinimumHeight(30)
        self.user_prompt_ipt.setMinimumWidth(250)
        self.user_prompt_ipt.textChanged.connect(self.update_custom_prompt)
        # Placeholder
        self.custom_prompt = "" # [ts]
        self.control_layout.addWidget(QLabel("附加提示词:"), 3, 0)
        self.control_layout.addWidget(self.user_prompt_ipt, 3, 1)
        
        # Execute Task Mode (Current/All)
        self.execute_mode = "current" # alt "all"
                
        # Execute Task
        self.execute_btn = ExecutableButton("运行当前") # Alt "运行全部"
        self.execute_btn.clicked.connect(self.execute_current_task)
        self.execute_btn.setStyleSheet("""background-color: #4CAF50; 
                                          color: white; 
                                          font-weight: bold; 
                                          border-radius: 5px;
                                       """)
        # Bind if right click switch mode
        self.execute_btn.on_right_click = self.switch_exec_mode
        self.control_layout.addWidget(self.execute_btn, 4, 0, 1, 2)
        
        self.right_layout.addLayout(self.control_layout)
        self.right_layout.addStretch()
        self.content_layout.addWidget(self.right_widget)
        
        self.main_layout.addLayout(self.content_layout)
        
        # Bottom: Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.main_layout.addWidget(self.progress_bar)

        # Bottom Progress Bar Value
        self.current_progress_bar_val = 0 # [ts]
        
        #################################################################
        #
        # Task and History

        # Note, task should be assigned after every [ts] is preallocated

        # Current task data
        self.current_task_id = 0       # a valid task starts from 0
        self.cumulative_task_id = 0    # by default, the 1st task
        self.current_task_name = "任务0"
        self.current_task_flag = -1    # Task flag: -1 for notready, 0 for ready but unfinished, 1 for completed
        
        # Set current task to visable table
        self.history_list.addItem(self.current_task_name)
        self.history_list.setCurrentItem(self.history_list.item(0))  # Select the current one
        
        # Task History Data
        self.tasks_history: dict = {}  # Dict of task dicts. Key: name, val: task dict
        #                                A task dict is {'name': str, 'id': int, 'flag': int, 'attr': {...}}
        # Only stores NON-current data, current data will be removed

        #################################################################
        #
        # Appearance
        
        # Current theme
        self.current_theme = "dark" # Default dark, why? I personally like it ~
        if theme in ("dark", "light"):
            self.current_theme = theme
        
        # Apply theme now
        self.apply_theme()
        
        #################################################################
        #
        # Algorithm
        
        # Set global seed
        random.seed(random.SystemRandom().randint(0, 2**32-1)) # set random seed
        
        #################################################################
        #
        # Special Magic
        
        # Magic Ability
        # Layer: "all", "gal", None
        self.magic_enable_magic = enable_magic
        if self.magic_enable_magic is not None:
            if self.magic_enable_magic.lower() not in ("all", "gal"):
                self.magic_enable_magic = None
            else:
                self.magic_enable_magic = self.magic_enable_magic.lower()
        
        #################################################################
        #
        # Inference Related
        
        # Automatically set inference device or use manual input
        self.inference_device = torch.device(inference_device) if isinstance(inference_device, str) else inference_device
        if self.inference_device is None:
            self.inference_device = self._inference_device_env()
            # Note, to get a string, use self.inference_env.type
        else:
            self.inference_device = self._inference_device_best_avial_chk(self.inference_device.type)
            
        self.inference_support_bf16 = self._inference_dtype_bf16_support()

        # Automatically set inference quant or use manual input
        self.inference_quant = quant
        if self.inference_quant is None:
            if self.inference_device.type.startswith("cuda"):
                self.inference_quant = self._inference_best_avail_quant()

        # Is model busy
        self.model_isbusy = False          # Synchron with the following at the same time

        # Model is currently on which task? task name
        self.model_running_on_task_name = None  # None if not running

        # Model has been assigned to task of 
        self.model_assigned_task_list_ofnames = [] # Empty if not running
        
        ################################################################
        #
        # Finalize the form
        
        # Menu Bar
        self.setup_menu_bar() # Must be after magic
        
        # New Title
        self.update_title() # Must be after than model_isbusy
        
        ################################################################
        #
        # Load Backend
        
        # Create a model instance
        if not DSOCR_DEBUG:                
            # Initialize model wrapper
            self.model = DeepSeekOCRModel(model_path=DSOCR_UI_DEFAULT_MODELPATH, 
                                          device=self.inference_device, quant=self.inference_quant, 
                                          dtype = "bfloat16" if self.inference_support_bf16 else "float16")

    # Geretate standard hash id
    def _std_hash(self, a: str | bytes) -> str:
        if a:
            if isinstance(a, str):
                return hashlib.sha256(a.encode("utf-8")).hexdigest()
            elif isinstance(a, bytes):
                return hashlib.sha256(a).hexdigest()
            else:
                return hashlib.sha256(str(a).encode("utf8")).hexdigest()
        else:
            return ""

    # Generate hash for a given image on disk
    def _std_hash_filepath(self, path: str) -> str | None:
        if path is None or os.path.exists(path) == False:
            return None
        buffer = b""
        with open(path, "rb") as f:
            f.write(buffer)
        return self._std_hash(buffer)

    # Detect Inference Environment if CUDA is available
    def _inference_device_env(self) -> str:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Manually set inference environment (will check availbility)
    def _inference_device_best_avial_chk(self, g_dtype):
        
        if isinstance(g_dtype, torch.device):
            g_dtype = g_dtype.type
            
        if g_dtype.strip().lower() == "cpu":
            return torch.device("cpu")
        else:
            if torch.cuda.is_available():
                return torch.device(g_dtype)
            else:
                return torch.device("cpu")            

    # Detect Having BFLOAT16 Support
    def _inference_dtype_bf16_support(self, device = None) -> bool:
        if device is None or device == "cuda":
            try:
                return torch.cuda._check_bf16_tensor_supported("cuda")
            except:
                return False
        else:
            try:
                return torch.cuda._check_bf16_tensor_supported(device)
            except:
                return False
    
    # Detect Best Available Quant
    def _inference_best_avail_quant(self) -> str | None:
        if torch.cuda.is_available():
             best = "nf4" if supports_bnb_nf4_() else None
             if best is None:
                 best = "bnb_8bit" if supports_bnb_8bit_() else None
                 
             # Well, below that is not actually used in real quantization
             if best is None:
                 best = "bfloat16" if self._inference_dtype_bf16_support() else "float16"
             return best
        else:
            return None
    
    # Task Mapping
    def _task_mapping(self, task: str, *, inputs: str = ""):
        # "自由OCR": "Free OCR"
        # "图片转文字": "Standard OCR"
        # "图片转Markdown": "Convert to Markdown"
        # "图表解读": "Parse Figure"
        # "图片描述": "Describe Image in Chinese"
        # "全图找物": "Locate Object by Reference"
        # "全图翻译": "Translate into"
        # "全图问答": "Q&A Chinese"
        # "自定义": Custom Mode
        if task == "自由OCR":
            return "Free OCR"
        elif task == "图片转文字":
            return "Standard OCR"
        elif task == "图片转Markdown":
            return "Convert to Markdown"
        elif task == "图表解读":
            return "Parse Figure"
        elif task == "图片描述":
            return "Describe Image in Chinese"
        elif task == "全图找物":
            return "Locate Object by Reference"
        elif task == "全图翻译":
            return "Translate into"
        elif task == "全图问答":
            return "Q&A Chinese"
        elif task == "自定义":
            return inputs
        else:
            # by default: Free OCR
            return "Free OCR"
    
    # Background Image
    def set_background_image(self, bg_path):
        if os.path.exists(bg_path):
            self.background_path = bg_path
            palette = QPalette()
            pixmap = QPixmap(self.background_path).scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,   # Zoom up instead of crop
                Qt.TransformationMode.SmoothTransformation
            )
            palette.setBrush(QPalette.ColorRole.Window, QBrush(pixmap))
            self.central_widget.setPalette(palette)
            self.central_widget.setAutoFillBackground(True)
        else:
            pass
            
    # Menu Bar
    def setup_menu_bar(self):
        
        # Initialize menubar
        self.menubar = self.menuBar()
        
        # File Actions
        self.file_menu = self.menubar.addMenu("文件")
        self.file_menu.addAction("设置工作目录", self.set_working_directory)
        self.file_menu.addAction("浏览工作目录", self.open_working_directory)
        self.file_menu.addAction("浏览缓存目录", self.open_working_tmp_directory)
        self.file_menu.addAction("浏览镜像目录", self.open_last_img_saving_directory)
        self.file_menu.addAction("浏览当前图片", self.open_current_image_dir)
        self.file_menu.addAction("退出", self.close)

        # Setting Actions
        self.settings_menu = self.menubar.addMenu("设置")
        self.settings_theme_action = self.settings_menu.addAction("界面主题")
        self.settings_theme_action.triggered.connect(self.open_theme_settings)
        self.settings_infenv_action = self.settings_menu.addAction("推理环境")
        self.settings_infenv_action.triggered.connect(self.open_infenv_settings)
        
        # Image Actions
        self.wimg_menu = self.menubar.addMenu("镜像")
        self.wimg_menu.addAction("保存工作镜像", self.save_task_img)
        self.wimg_menu.addAction("加载工作镜像", self.reload_task_img)
        
        # Magic Actions
        if self.magic_enable_magic is not None:
            self.magicaaa_menu = self.menubar.addMenu("魔法")
            if self.magic_enable_magic in ("all", "gal"):
                # Galgame magic
                self.magicaaa_menu.addAction("随机Galgame", lambda: self.magic_random_gal_avif(False, None))
                self.magicaaa_menu.addAction("继续探索此游戏", lambda: self.magic_random_gal_avif(True, None))
                self.magicaaa_menu.addAction("启用不安全内容", lambda: self.magic_random_gal_avif(False, True))
                self.magicaaa_menu.addAction("禁用不安全内容", lambda: self.magic_random_gal_avif(False, False))
                          
        # About Actions
        self.about_menu = self.menubar.addMenu("关于")
        self.about_menu.addAction("模型", self.show_modelpage)
        self.about_menu.addAction("程序", self.show_github)
        self.about_menu.addAction("作者", self.show_bilibili)
        self.about_menu.addAction("基础用法", self.show_basicusage)
        self.about_menu.addAction(f"版本 {DSOCR_UI_DEFAULT_GUIVER}", self.do_nothing)
        
    # Do Nothing placeholder
    def do_nothing(self):
        return
    
    # Count Ready Tasks
    def count_ready_tasks(self) -> int:
        n = 0
        for i in range(self.history_list.count()):
            item = self.history_list.item(i)
            state_int = self.history_list._get_item_state_int(item)
            # For each ready, we process it
            if state_int == 0:
                n += 1
        return n

    # Switch Executable Mode
    def switch_exec_mode(self):
        # current <--> all
        if self.execute_mode == "current":
            
            # use history list to deduce the number of readys
            n = self.count_ready_tasks()
            # set
            self.execute_mode = "all"
            self.execute_btn.setText(f"运行全部 ({n})")
            # bind
            self.execute_btn.clicked.disconnect()
            self.execute_btn.clicked.connect(self.execute_all_selected_tasks)
            
        elif self.execute_mode == "all":
            self.execute_mode = "current"
            self.execute_btn.setText("运行当前")
            self.execute_btn.clicked.disconnect()
            self.execute_btn.clicked.connect(self.execute_current_task)
        return
    
    # Update Executable Button n
    def update_exec_button_n(self):
        # This will be called after any state change
        if self.execute_mode == "all":
            
            # use this flag and history flags
            n = 1 if self.current_task_flag == 0 else 0
            for k in self.tasks_history.keys():
                # To avoid legacy version of current name stored
                if k != self.current_task_name:
                    d = self.tasks_history[k]
                    if d.get("flag", None) == 0:
                        n += 1
            self.execute_btn.setText(f"运行全部 ({n})")
    
    # Show DS OCR Model Page
    def show_modelpage(self):
        url = r"https://huggingface.co/deepseek-ai/DeepSeek-OCR"
        QDesktopServices.openUrl(QUrl(url))

    # Show Github Homepage
    def show_github(self):
        url = r"https://github.com/dof-studio/DSOCR"
        QDesktopServices.openUrl(QUrl(url))
        
    # Show Bilibili Homepage
    def show_bilibili(self):
        url = r"https://space.bilibili.com/303266889"
        QDesktopServices.openUrl(QUrl(url))
        
    # Show basic usage of this sftwr
    def show_basicusage(self):
        QMessageBox.warning(None, "Unimplemented", "敬请期待...        ")
        
    # Convert img to png using PIL
    def convert_image_to_png(self, img_path) -> str | None:
        if img_path and os.path.exists(img_path):
            try:
                img = Image.open(img_path)
                spl = img_path.split(".")
                # Replace suffix to png
                new_path = ".".join([c if i != len(spl) - 1 else "png" for i, c in enumerate(spl)])
                img.save(new_path, format="PNG")
                return new_path
            except:
                # Silent Mode
                return None
        return None
        
    # Get a random galgame banner image
    def magic_random_gal_avif(self, i_like_it_mode_ = False, enable_nsfw = None):
        
        # If magic/galgame magic is disabled, return
        if self.magic_enable_magic is None or self.magic_enable_magic not in ("all", "gal"):
            return
        
        # If ENABLE nsfw is first called? We will warn everybody that NSFW Content may have
        if enable_nsfw == True and getattr(self, "magic_random_gal_avif_nsfw_fbi_warning", None) is None:
            reply = QMessageBox.question(None, "警告", "随机Galgame图片来自于网络，启用不安全内容后可能包含过激和限制内容(NSFW)。\n请确认您已年满18周岁，并确认图片由您自己获取:",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                # Enabling NSFW
                self.magic_random_gal_avif_nsfw_fbi_warning = "passed"
                self.magic_random_gal_avif_sel_sfwonly = False
                return # Successful but do not get it
            else:
                return # Do nothing we can't provide this for people under 18  
        elif enable_nsfw == True:
            self.magic_random_gal_avif_sel_sfwonly = False
            return
            # If already had, we think user has confirmed   
        
        # If DISABLE nsfw is called, just disable it
        if enable_nsfw == False:
            self.magic_random_gal_avif_sel_sfwonly = True
            return
        
        # We will create a flag to identify if user only wants non-NSFW content
        if getattr(self, "magic_random_gal_avif_sel_sfwonly", None) is None:
            self.magic_random_gal_avif_sel_sfwonly = True
            # If True, filter content with \"contentLimit\":\"nsfw\"
        
        # i_like_it_mode_ : If enabled and using unique id mode, I will try to randomly get the same image from the same game
        #                   By default, disabled
        #                   See self.magic_random_gal_avif_prevlist if it exists
        if getattr(self, "magic_random_gal_avif_prevlist", None) is not None and i_like_it_mode_ == False:
            # New game, set it to None
            self.magic_random_gal_avif_prevlist = None
        
        # Crypto Utils to decode patchId
        def _cached_patchid_decipher(cipher):

            if cipher is None:
                return None

            # 1) Construct public key
            if getattr(self, "magic_random_gal_avif_patchid_rsa_key", None) is None:
                try:
                    path_pem = os.path.join(DSOCR_UI_DEFAULT_RSPATH, "key", "crypto_rsa_key.publ")
                    publ_pem = load(path_pem)   
                    self.magic_random_gal_avif_patchid_rsa_key = import_rsa_key(publ_pem)
                except:
                    self.magic_random_gal_avif_patchid_mapping = {} # Nullify, avoid using corrupted
                    return None # failed
                
            # 2) Check signature
            if getattr(self, "magic_random_gal_avif_pathid_mgg_chk", None) is None:
                try:
                    mgg_path = os.path.join(DSOCR_UI_DEFAULT_RSPATH, "sp", "mgg.pkl")
                    sig_path = os.path.join(DSOCR_UI_DEFAULT_RSPATH, "sp", "sig.pkl")
                    b = b""
                    with open(mgg_path, "rb") as f:
                        b = f.read()     
                    s = load(sig_path)
                    chksum_de = rsa_public_decrypt_raw(self.magic_random_gal_avif_patchid_rsa_key, s)
                    chksum_re = blake2b_derive_key(b, 32)
                    if chksum_de != chksum_re:
                        raise RuntimeError("")
                    self.magic_random_gal_avif_pathid_mgg_chk = chksum_re
                except:
                    self.magic_random_gal_avif_patchid_mapping = {} # Nullify, avoid using corrupted
                    return None
            
            # 3) Load AES cipher and restore original
            if getattr(self, "magic_random_gal_avif_patchid_aes_plain", None) is None:
                try:
                    path_aes = os.path.join(DSOCR_UI_DEFAULT_RSPATH, "key", "crypto_passkey.ciph")
                    encrypted_by_private = load(path_aes)
                    self.magic_random_gal_avif_patchid_aes_plain = rsa_public_decrypt_raw(self.magic_random_gal_avif_patchid_rsa_key, encrypted_by_private)
                except:
                    self.magic_random_gal_avif_patchid_mapping = {} # Nullify, avoid using corrupted
                    return None

            # 4) Decrypt content
            try:
                v, cp = cipher
                pl = aes_cbc_decrypt(self.magic_random_gal_avif_patchid_aes_plain, v, cp).decode("utf-8")
                return pl
            except:
                return None
        
        # Cached patchId mapping
        def _cached_patchid_mapping(rand) -> str | None:
            # If we can load the patch id
            if getattr(self, "magic_random_gal_avif_patchid_mapping", None) is None:
                path = os.path.join(DSOCR_UI_DEFAULT_RSPATH, "sp", "mgg.pkl")
                if os.path.exists(path):
                    try:
                        d = load(path, kompress=lzma)
                        self.magic_random_gal_avif_patchid_mapping = d
                        return _cached_patchid_decipher(self.magic_random_gal_avif_patchid_mapping.get(rand, None)).strip()
                    except:
                        pass
                # Invalid dict, we give it an empty dict
                self.magic_random_gal_avif_patchid_mapping = {}
                return None
            else:
                try:
                    return _cached_patchid_decipher(self.magic_random_gal_avif_patchid_mapping.get(rand, None)).strip()
                except:
                    return None
            
        # Make header
        @lru_cache
        def _make_header():
            headers = {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                "Accept-Encoding": "gzip, deflate, br, zstd",
                "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
                "Cache-Control": "max-age=0",
                "Priority": "u=0, i",
                "Sec-Ch-Ua": '"Google Chrome";v="125", "Chromium";v="125", "Not.A/Brand";v="24"',
                "Sec-Ch-Ua-Mobile": "?0",
                "Sec-Ch-Ua-Platform": '"Windows"',
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Upgrade-Insecure-Requests": "1",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
            }
            return headers
            
        # Cached redirected new website patchid base
        def _cached_redirected_patchid_base(zero):
            if zero != 0:
                # DOES NOT WORK IF NOT ZERO
                return None
            elif getattr(self, "magic_random_gal_pathid_base", None) is None:
                # trail, using .io but may not necessarily be this
                try:
                    # @MAGIC
                    # @CHANGE
                    # We assume .us is currently okay and stable
                    # This may change, so provice an alternative approach
                    try:
                        url = "https://touchgal.us/api/patch/resource?patchId=1" # will redirect to new homepage
                        response = requests.get(url, headers=_make_header(), timeout=3)
                        new_url = None
                        if response.status_code == 200:
                            new_url = response.url
                            # extract https://*/
                            new_url = "https://" + new_url.split("https://")[1].split("/")[0]
                        else:
                            raise RuntimeError("Access to root is denied")
                    except:
                        url = "https://huggingface.co/dof-studio/urle/resolve/main/DSOCR.urle"
                        response = requests.get(url, headers=_make_header())
                        new_url = None
                        if response.status_code == 200:
                            new_url = response.text.strip()
                            # extract https://*/
                            new_url = "https://" + new_url.split("https://")[1].split("/")[0]
                    # Set it and we can use it in the future
                    self.magic_random_gal_pathid_base = new_url
                    return new_url + "/api/patch/resource?patchId="
                except:
                    return None
            else:
                # Already cached, then we use it
                return self.magic_random_gal_pathid_base + "/api/patch/resource?patchId="

        # Cached URL getter (Generic)
        @lru_cache
        def _cached_url_avif_getter(url: str) -> str | None:
            
            hsh = self._std_hash(str(url))
            tmpfolder = os.path.join(self.current_wd, "tmp")
            dest = os.path.join(tmpfolder, hsh + ".avif")
                                    
            # Request
            try:
                response = requests.get(url, headers=_make_header(), stream=True, timeout=15)
                if response.status_code == 200:
                    with open(dest, "wb") as f:
                        for chunk in response.iter_content(chunk_size=65536):
                            if chunk:
                                f.write(chunk)
                    # If success, link img to the path
                    return dest
                else:
                    # Fail return None
                    return None
            except Exception as e:
                # If encountered any, return None
                return None
        
        # Cached Text getter (Generic)
        @lru_cache
        def _cached_url_text_getter(url: str) -> str | None:
                                    
            # Request
            try:
                response = requests.get(url, headers=_make_header(), timeout=10)
                if response.status_code == 200:
                    return response.text
                else:
                    # Fail return None
                    return None
            except Exception as e:
                # If encountered any, return None
                return None
        
        # Get game banner (folder image)
        def download_random_banner(max_retry = 5) -> Tuple[str | None, int]:
            
            # Path or None
            
            # Random number between 1 to 13000 (well, they may add more but as of now, 14k is enough)
            rand = random.randint(1, 13000)
            
            # Cached Banner getter
            @lru_cache
            def _cached_banner_getter(rand) -> str:
                
                base = "https://cloud.touchgaloss.com/patch/{rand}/banner/banner.avif"
                url = base.format(rand=rand)
                return _cached_url_avif_getter(url)
            
            # Cached Capture getter
            
            
            # while false, download it
            imgpth = None
            count = 0
            while imgpth is None:
                
                # Exceeding? Damn it
                if count >= max_retry:
                    return None, None
                
                # Process file and dest
                count += 1
                
                
                # Try get it
                rand = random.randint(1, 13000)
                imgpth = _cached_banner_getter(rand)
                if imgpth is not None:
                    break
        
            # If got you, then return it else return None
            return imgpth, rand
        
        # Get random capture image
        def download_random_capture(max_retry = 5) -> Tuple[str | None, int]:
           
            # Path or None
            
            # Random number between 1 to 13000 (well, they may add more but as of now, 14k is enough)
            rand = random.randint(1, 13000)
            
            # Parse the unique id and try get the image list
            def _get_image_list(rand) -> list:
                
                # Use cached unique id is possible
                uniqueid = _cached_patchid_mapping(rand)
                if uniqueid is not None:
                    hpurl = _cached_redirected_patchid_base(zero = 0).replace("/api/patch/resource?patchId=", "/") + uniqueid
                
                # Failed, then auto search online
                else:
                    pathidurl = _cached_redirected_patchid_base(zero = 0) + str(rand)
                    metadata = _cached_url_text_getter(url = pathidurl)
                    hpurl = None
                    if metadata:
                        try:
                            metadata = json.loads(metadata)[0]
                            uniqueid = metadata.get("uniqueId", None).strip()
                            if uniqueid:
                                hpurl = _cached_redirected_patchid_base(zero = 0).replace("/api/patch/resource?patchId=", "/") + uniqueid
                        except:
                            return []
                    else:
                        return []
                
                # If survive, request for the main page
                hpdata = _cached_url_text_getter(url = hpurl)
                if not hpdata:
                    # print("can't get text")
                    return []
                
                # See if passed the filter of non-NSFW
                if self.magic_random_gal_avif_sel_sfwonly == True:
                    if hpdata.find('nsfw') != -1:
                        return []
                
                # Get avif urls
                avif_urls = re.findall(r'https?://[^\s"\'<>]+\.avif', hpdata)
                if len(avif_urls) > 0:
                    return avif_urls
                else:
                    return []
                
            # while false, download it
            imgpth = None
            count = 0
            while imgpth is None:
                
                # Exceeding? Damn it
                if count >= max_retry:
                    return None, None
                
                # Process file and dest
                count += 1
                
                # Try get game list
                rand = random.randint(1, 13000)
                imglist = [s.strip() for s in _get_image_list(rand)]
                if len(imglist) > 0:
                    # Set current random game but drop user banner (same every time)
                    imglist = list(set(imglist) - set(['https://cloud.touchgaloss.com/user/avatar/user_1/avatar-mini.avif']))
                    self.magic_random_gal_avif_prevlist = imglist
                else:
                    continue
                
                # Well, we don't actually randomly choose, we push the recent one to the end
                if len(imglist) >= 2:
                    rand = random.randint(0, len(imglist)-2) # Will never choose the prev one
                    others = set(range(len(imglist))) - set([rand])
                    newseq = list(others) + [rand]
                    imgrandom = imglist[rand]
                    # Apply the new sequence
                    imglist = make_array(imglist)[newseq].tolist()
                    self.magic_random_gal_avif_prevlist = imglist
                else:
                    imgrandom = imglist[0]
                
                # Try get it
                imgpth = _cached_url_avif_getter(imgrandom)
                if imgpth is not None:
                    break
                else:
                    # Try another one
                    imgrandom = imglist[random.randint(0, len(imglist)-1)]
                    imgpth = _cached_url_avif_getter(imgrandom)
                    if imgpth is not None:
                        break
                    else:
                        continue
        
            # If got you, then return it else return None
            return imgpth, rand
        
        # Get random image in this game (current game, i_like_it_mode)
        def download_random_capture_ilikeit(max_retry = 5, i_like_it_mode_ = False) -> Tuple[str | None, int]:
            
            # If not saved or not i_like_it_mode_, fall back
            if getattr(self, "magic_random_gal_avif_prevlist", None) is None or i_like_it_mode_ == False:
                return download_random_capture(max_retry)
            
            # while false, download it
            imgpth = None
            count = 0
            while imgpth is None:
                
                # Exceeding? Damn it
                if count >= max_retry:
                    return None, None
                
                # Process file and dest
                count += 1
                
                # Cool, let's randomlt choose one     
                imglist = self.magic_random_gal_avif_prevlist
                
                # Well, we don't actually randomly choose, we push the recent one to the end
                if len(imglist) >= 2:
                    rand = random.randint(0, len(imglist)-2) # Will never choose the prev one
                    others = set(range(len(imglist))) - set([rand])
                    newseq = list(others) + [rand]
                    imgrandom = imglist[rand]
                    # Apply the new sequence
                    imglist = make_array(imglist)[newseq].tolist()
                    self.magic_random_gal_avif_prevlist = imglist
                else:
                    imgrandom = imglist[0]
                
                # Try to get it 
                imgpth = _cached_url_avif_getter(imgrandom)
                if imgpth is not None:
                    break
        
            # If got you, then return it else return None
            return imgpth, None # the 2nd is not used...
        
        # Go
        # [Logic]:
        # If good, we use local cache and directly access the images randomly
        #   If ilikeit, then choose from the same game
        #   Else, we choose a random game from our pool
        # Else, we generate random number and try one by one
        #
        # Get the new pathid base
        pathid_base = _cached_redirected_patchid_base(zero=0)
        if pathid_base is None:
            # Fail to use the banner version
            # Run a random getter
            imgpth, num = download_random_banner(5)
        else:
            # If it is good, we can get whatever image we want
            # Use generic api (it will fall to original mode if not I LIKE IT)
            imgpth, num = download_random_capture_ilikeit(5, i_like_it_mode_)
            
        # Process the img
        if imgpth is None:
            # Error, too many times
            QMessageBox.warning(None, "错误", "由于网络或偶然问题无法加载随机图片，请稍后再试")
            return
        else:
            # Good. Now we convert avif to png
            imgpth = self.convert_image_to_png(imgpth)
            if imgpth:
                self.load_image(imgpth)
        
    # Set working directory
    def set_working_directory(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择工作目录", self.current_wd)
        if folder_path and os.path.exists(folder_path):
            if self.current_wd != folder_path:
                self.current_wd = folder_path

    # Open working directory
    def open_working_directory(self):
        if self.current_wd and os.path.exists(self.current_wd):
            # Open the folder
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.current_wd))
        else:
            # What? No valid working directory? damn, create it!
            self.current_wd = DSOCR_UI_DEFAULT_WDPATH
            if os.path.exists(self.current_wd) == False:
                os.makedirs(self.current_wd, exist_ok=True)

    # Open working tmp directory
    def open_working_tmp_directory(self):
        tmp = os.path.join(self.current_wd, "tmp")
        if tmp and os.path.exists(tmp):
            # Open the folder
            QDesktopServices.openUrl(QUrl.fromLocalFile(tmp))
        else:
            # What? No tmp? Damn it. Create one
            if tmp:
                os.makedirs(tmp, exist_ok=True)     
            
    # Open last time img saving directory if any
    def open_last_img_saving_directory(self):

        if self.last_task_img_folder is None:
            # Haven't saved
            QMessageBox.warning(None, "错误", "无法打开最后镜像存储目录\n当前实例下没有存储镜像，请先选择存储镜像")
        else:
            p = Path(self.last_task_img_folder).resolve()
            # Open the folder
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))

    # Open directory of the current selected image
    def open_current_image_dir(self):
        if self.current_image_path and os.path.exists(self.current_image_path):
            # Resolve path
            p = Path(self.current_image_path).resolve()
            if p.is_dir():
                p = str(p)
            else:
                p = str(p.parent)
            # Open the folder
            QDesktopServices.openUrl(QUrl.fromLocalFile(p))
        else:
            # No image or deleted? Warning
            QMessageBox.warning(None, "错误", "无法打开所在目录\n当前任务没有加载有效的图片，或者图片已经失效")
    
    # Open Theme Setting
    def open_theme_settings(self):
        dialog = Theme_SettingsDialog(self)
        dialog.exec()
    
    # Open Inference Env Setting
    def open_infenv_settings(self):
        dialog = InferEnv_SettingsDialog(self)
        dialog.exec()
    
    # Resize Event: reset current theme
    def resizeEvent(self, event):
        # When resized, apply new theme
        self.set_theme(self.current_theme)
        super().resizeEvent(event) 
    
    # Set Theme
    def set_theme(self, theme):
        
        # Set theme
        self.current_theme = theme
        self.apply_theme()
        
        # Set Task List theme
        self.history_list.reset_theme(theme)
        
        # Update Title
        self.update_title()
    
    # Apply Theme
    def apply_theme(self):
        
        # DARK THEME
        if self.current_theme == "dark":
            
            # Set Style
            self.setStyleSheet("""   
                QMainWindow {
                    background-color: #2b2b2b;
                    color: #ffffff;
                }
                QLabel {
                    color: #ffffff;
                }
                QImageLabel {
                    background-color: rgba(64,64,64,0.6);
                    color: #ffffff;
                    border: 1px solid #555;
                    border-radius: 5px;
                }
                QPushButton {
                    background-color: #404040;
                    border: 1px solid #555;
                    color: #ffffff;
                    padding: 8px;
                    border-radius: 6px;
                }
                QPushButton:hover {
                    background-color: rgba(64,64,64,0.8);
                }
                QComboBox {
                    background-color: rgba(64,64,64,0.8);
                    border: 1px solid #555;
                    color: #ffffff;
                    padding: 4px;
                    border-radius: 6px;
                }
                QTextEdit, QListWidget {
                    background-color: rgba(64,64,64,0.8);
                    border: 1px solid #555;
                    color: #ffffff;
                    border-radius: 6px;
                }
                QLineEdit {
                    background-color: rgba(64,64,64,0.8);
                    border: 1px solid #555;
                    color: #ffffff;
                    border-radius: 6px;
                }
                QMenu {
                    background-color: rgba(64,64,64,0.8);
                    border: 1px solid #555;
                    border-radius: 6px;
                    color: white;
                }
                QMenu::item:selected {
                    background: rgba(100,100,100,0.8);
                    border-radius: 6px;
                    color: white;
                }
                QMenuBar {
                    background-color: rgba(64,64,64,0.8);
                    border: 1px solid #555;
                    color: white;
                }
                QMenuBar::item:selected {
                    background: rgba(100,100,100,0.8);
                    border-radius: 6px;
                    color: white;
                }
                QProgressBar {
                    background-color: rgba(64,64,64,0.8);
                    border: 1px solid #555;
                    color: #ffffff;
                    border-radius: 6px;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #4CAF50;
                    border: 1px solid #ccc; 
                    border-radius: 6px;
                }
            """)
            # Frosted glass effect simulation with semi-transparency
            for widget in [self.centralWidget(), self.image_label, self.current_output_text]:
                # effect = QGraphicsBlurEffect()
                # <Optional, blur>
                # effect.setBlurRadius(2)
                # widget.setGraphicsEffect(effect)
                pass
        
            # Set Background Image
            self.set_background_image(DSOCR_UI_DEFAULT_LGPATH)
        
        # LIGHT THEME
        else: 
            
            # Set Style
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #f5f5f5;
                    color: #333;
                }
                QLabel {
                    color: #333;
                }
                QImageLabel{
                    background-color: rgba(255,255,255,0.6);
                    color: #333;
                    border: 1px solid #ccc; 
                    border-radius: 6px;
                }
                QPushButton {
                    background-color: #e0e0e0;
                    border: 1px solid #ccc;
                    padding: 8px;
                    border-radius: 6px;
                }
                QPushButton:hover {
                    background-color: rgba(255,255,255,0.8);
                }
                QPushButton:disabled {
                    background-color: rgba(255,255,255,0.8);
                    color: #999;
                }
                QComboBox {
                    border: 1px solid #ccc;
                    padding: 6px;
                    border-radius: 6px;
                }
                QTextEdit, QListWidget {
                    border: 1px solid #ccc;
                    border-radius: 6px;
                    background-color: rgba(255,255,255,0.8);
                }
                QLineEdit {
                    background: rgba(255,255,255,0.8);
                    border: 1px solid #ccc;
                    border-radius: 4px;
                }
                QMenu {
                    background: rgba(255,255,255,0.8);
                    border: 1px solid #ccc;
                    border-radius: 6px;
                    color: black; 
                }
                QMenu::item:selected {
                    background: rgba(0,120,215,0.2);
                    border-radius: 6px;
                    color: black;
                }
                QMenuBar {
                    background: rgba(255,255,255,0.8);
                    border: 1px solid #ccc;
                    color: black; 
                }
                QMenuBar::item:selected {
                    background: rgba(0,120,215,0.2);
                    border-radius: 6px;
                    color: black;
                }
                QProgressBar {
                    background: rgba(255,255,255,0.8);
                    border: 1px solid #ccc;
                    border-radius: 6px;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #4CAF50;
                    border: 1px solid #ccc; 
                    border-radius: 6px;
                }
            """)
            
            # Frosted glass effect simulation with semi-transparency
            for widget in [self.centralWidget(), self.image_label, self.current_output_text]:
                # effect = QGraphicsBlurEffect()
                # <Optional, blur>
                # effect.setBlurRadius(2)
                # widget.setGraphicsEffect(effect)
                pass
            
            # Set Background Image
            self.set_background_image(DSOCR_UI_DEFAULT_BGPATH)
    
    # Update Inference Env
    def update_infenv(self, device: str, quant: str):

        # If model is busy, simply return 
        if self.model_isbusy:
            return
        # quant can be nf4, bnb_8bit, bfloat16, float16, None (for CPU), None (Auto deduction mode)
        
        # Automatically set inference device or use manual input
        self.inference_device = device
        if self.inference_device is None:
            self.inference_device = self._inference_device_env()
            # Note, to get a string, use self.inference_env.type
        else:
            self.inference_device = self._inference_device_best_avial_chk(self.inference_device.type)
            
        self.inference_support_bf16 = self._inference_dtype_bf16_support()

        # Automatically set inference quant or use manual input
        self.inference_quant = quant
        if self.inference_quant is None:
            if self.inference_device.type.startswith("cuda"):
                self.inference_quant = self._inference_best_avail_quant()

        # Move model
        if not DSOCR_DEBUG:    
            
            # If push to cpu, then directly move
            if self.inference_device.type.lower() == "cpu":
                self.model.to_cpu_inplace()
            
            # Else, reload
            else:
                self.model.reload_model(device=self.inference_device, quant=self.inference_quant, 
                                        dtype = "bfloat16" if self.inference_support_bf16 else "float16")

        # Finally, update title
        self.update_title()

    # Title Env Helper
    def _title_env_helper(self):
        base = self.inference_device.type.upper()
              
        # Inference Env
        if self.inference_quant == "nf4":
            base += " NF4"
        elif self.inference_quant == "bnb_8bit":
            base += " INT8"
        elif self.inference_quant == "bfloat16":
            base += " BF16"
        elif self.inference_quant == "float16":
            base += " FP16"
        elif self.inference_quant == "float32":
            base += " FP32"
        else:
            pass
        
        # Style
        if self.current_theme == "light":
            base += " - 澄空主题"
        elif self.current_theme == "dark":
            base += " - 静海主题"
        
        return base

    # Update Title
    def update_title(self):
        self.setWindowTitle(f"DS OCR 图片转文字工具 ({self._title_env_helper()}){' 正忙 ···' if self.model_isbusy else ''}")
            
    # Control Panel Changed Base
    def _control_panel_changed_base(self, *, without_touch_state=False):
        # Note, state changed. By default we change to 0 if an image 
        if without_touch_state == False:
            if self.has_current_image():
                self.change_current_task_state(0, sendtolist=True) # actively

    # Update Custom Prompt
    def update_custom_prompt(self, text, *, without_touch_state=False):
        if text.strip():
            self.custom_prompt = text.strip()
            # Control Panel Changed
            self._control_panel_changed_base(without_touch_state=without_touch_state)
        else:
            pass
    
    # Set new current image and roll back to last if possible
    def _update_current_image_path(self, new_path, *, coersive=False):
        if (new_path and os.path.exists(new_path)) or coersive:
            self.last_image_path = self.current_image_path
            self.current_image_path = new_path
        return
      
    # Callback, has current image (bool, True if current image is set)
    def has_current_image(self) -> bool:
        if self.current_image_path and os.path.exists(self.current_image_path):
            return True
        else:
            return False
        
    # Callback, does an element have current image (bool)
    def has_certain_elem_current_image(self, item) -> bool:
        
        # If this item is current item, fall back to current
        name = item.text()
        if name == self.current_task_name:
            return self.has_current_image()
        
        # If can't find name in hist then False
        if name not in self.tasks_history.keys():
            return False
        
        # Try to get a reference of task list
        task_list_ref = self.tasks_history.get(name, {})
        if task_list_ref.get("attr", None) is None:
            return False
        if task_list_ref["attr"].get("current_image_bytes", None) is not None:
            return True
        else:
            return False
      
    # API Load an Image
    def load_image(self, path_or_pil, *, without_touch_state=False):
        
        # If None then return
        if path_or_pil is None:
            return
        
        # If a path
        if os.path.exists(path_or_pil):
            pixmap = QPixmap(path_or_pil)
            scaled_pixmap = pixmap.scaled(3000, 4000, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.image_label.setImage(scaled_pixmap)
            self.image_label.setText("")
            self._update_current_image_path(path_or_pil)
            self.copy_image_btn.setEnabled(True)
            self.clr_image_btn.setEnabled(True)
            # If having a valid last image, then true, else false
            if self.last_image_path is not None and os.path.exists(self.last_image_path):
                self.rollback_image_btn.setEnabled(True)
            else:
                self.rollback_image_btn.setEnabled(False)
            # Change flag to 0 once got changed
            if without_touch_state == False:
                self.change_current_task_state(0, sendtolist=True) # actively
            
        # Else if a pil (NOT IMPLEMENTED)
        else:
            pass
        # @TODO
        
    # Roll Back to Last Image (if any)
    def rollback_last_image(self, *, naive=True, without_touch_state=False):
        
        # If last image exists
        if self.last_image_path is not None and os.path.exists(self.last_image_path):
            # If Naive == False, then drop the current; else, just switch them
            if naive == False:
                image_path = self.last_image_path
                self.current_image_path = None
                self.load_image(image_path)
            else:
                self.load_image(self.last_image_path)
            # Change flag to 0 once got changed
            if without_touch_state == False:
                self.change_current_task_state(0, sendtolist=True) # actively
        
    # Try saving image in clipboard
    def _save_clipboard_image(self, folder_path: str) -> str:
        try:
            img = ImageGrab.grabclipboard()
            if img is None:
                return ""
            # Calculate hash as filename
            img_bytes = img.tobytes()
            img_hash = hashlib.sha256(img_bytes).hexdigest()
            # Save the image
            if not os.path.exists(folder_path):
                os.makedirs(folder_path, exist_ok=True)
            file_path = os.path.join(folder_path, f"{img_hash}.png")
            img.save(file_path, "PNG")
            return os.path.abspath(file_path)
        except:
            return ""
        
    # Upload Image with a Popup
    def upload_image(self, *, without_touch_state=False): 
        file_path, _ = QFileDialog.getOpenFileName(self, "选择并上传图片", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp *.dng)")
        if file_path:
            pixmap = QPixmap(file_path)
            if pixmap.isNull():
                QMessageBox.warning(None, "错误", "无效的图片文件.")
                return
            scaled_pixmap = pixmap.scaled(3000, 4000, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.image_label.setImage(scaled_pixmap)
            self.image_label.setText("")
            self._update_current_image_path(file_path)
            # Enabling
            self.copy_image_btn.setEnabled(True)
            self.clr_image_btn.setEnabled(True)
            # If having a valid last image, then true, else false
            if self.last_image_path is not None and os.path.exists(self.last_image_path):
                self.rollback_image_btn.setEnabled(True)
            else:
                self.rollback_image_btn.setEnabled(False)
            # Change flag to 0 once got changed
            if without_touch_state == False:
                self.change_current_task_state(0, sendtolist=True) # actively
            
    # Copy Image from clipboard, save, and Load Image
    def load_image_from_clipboard(self):
        # Try get image
        tmpfolder = os.path.join(self.current_wd, "tmp")
        imgpath = self._save_clipboard_image(folder_path=tmpfolder)
        if imgpath:
            self.load_image(imgpath)
            # Other stuff will be done in load_image()
    
    # Clear Image (Just Clear Current)
    def clear_image(self, *, without_touch_state=False):
        self.image_label.clearImage()
        self._update_current_image_path(None, coersive=True)
        self.copy_image_btn.setEnabled(False)
        self.clr_image_btn.setEnabled(False)
        # If having a valid last image, then true, else false
        if self.last_image_path is not None and os.path.exists(self.last_image_path):
            self.rollback_image_btn.setEnabled(True)
        else:
            self.rollback_image_btn.setEnabled(False)
        # Change flag to -1 once got removed
        if without_touch_state == False:
            self.change_current_task_state(-1, sendtolist=True) # actively
    
    # Copy Image to Clipboard
    def copy_image_to_clipboard(self):
        if hasattr(self, 'current_image_path') and self.current_image_path:
            pixmap = QPixmap(self.current_image_path)
            QApplication.clipboard().setPixmap(pixmap)
        
    # Clear The Display Zone (return the original one)
    def clear_log(self, *, without_touch_state=False, granted_in_model_running=False):

        # If model is busy, denied
        if self.model_isbusy and granted_in_model_running == False:
            # If the clearing is granted in running/ at the beginnng, to clear the stage, it is allowed
            QMessageBox.warning(None, "错误", "模型正忙,请稍后执行清空操作")
            return
    
        # Clear the log output
        self.current_output_text.clear()
        # Change flag to 0 (ready) once got log removed
        if without_touch_state == False:
            # Return to 0 if having img, else -1
            if self.has_current_image():
                self.change_current_task_state(0, sendtolist=True) # actively
            else:
                self.change_current_task_state(-1, sendtolist=True) # actively
        
    # Append log kernel (NO state change inside)
    def append_log_(self, message: str, add_newline: bool = False, *, scroll_down = True):
        """
        Always insert the message at the end of the QTextEdit and scroll to the bottom
        """
        # Move the cursor to the end of the document
        cursor = self.current_output_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.current_output_text.setTextCursor(cursor)
    
        # Insert text (do not use append() to avoid automatic line wrapping)
        if add_newline:
            self.current_output_text.insertPlainText(message + "\n")
        else:
            self.current_output_text.insertPlainText(message)
    
        # Scroll down
        if scroll_down:
            self.current_output_text.verticalScrollBar().setValue(
                self.current_output_text.verticalScrollBar().maximum()
            )

    # Append Something New to Text Display Zone (emitable) (NO state change inside)
    @pyqtSlot(str) 
    def append_log(self, message):
        self.append_log_(message)
        
    # Copy Text Display Zone to Clipboard
    def copy_output_to_clipboard(self):
        text = self.current_output_text.toPlainText()
        # Drop EOF
        pest = re.compile(r'<[^>]*end[^>]*sentence[^>]*>', flags=re.IGNORECASE | re.UNICODE)
        text = pest.sub('', text)
        QApplication.clipboard().setText(text)
        
    # Copy Cleared Text to Clipboard
    def copy_cleared_output_to_clipboard(self):
        text = self.current_output_text.toPlainText()
        # Drop EOF, Drop Positional
        pest = re.compile(r'<[^>]*end[^>]*sentence[^>]*>', flags=re.IGNORECASE | re.UNICODE)
        text = pest.sub('', text)
        text = clear_positional_(text)
        QApplication.clipboard().setText(text)

    # Change state of the current task
    def change_current_task_state(self, new_state: int, *, sendtolist: bool = False):
        
        if new_state in (-1, 0, 1):
            # Change flag
            self.current_task_flag = new_state
            # Update button n if in all mode, auto deduction
            self.update_exec_button_n()

            # We never call reset current task here if sendtolist is True
            # since this is called by the outside main widget
            # But when and only when if this is called by change_certain_task_state()
            # sendtolist is by default False
            if sendtolist == False and new_state == -1:
                self.reset_current_task()
            
            # Special for current mode
            # If sendtolist is True, then user is using current widget to modify the state by operations
            # Then, we need to inform the TaskList to change appearance
            if sendtolist == True:
                item = self.history_list.currentItem()
                self.history_list.set_task_state_naive_noemit_(item, state=new_state)
        return
            
    # Change certain task state
    def change_certain_task_state(self, item, new_state: int, *, sendtolist: bool = False):

        # If this item points to current, then call 
        if item.text() == self.current_task_name:
            return self.change_current_task_state(new_state, sendtolist=sendtolist) 
        # This is invoked by HistoryList when sendtolit = False (always)
        # When sendtolist is true, maybe set by executor
        
        if new_state in (-1, 0, 1):
            name = item.text()
            # Fail if the same are same or old name not in hist
            if name not in self.tasks_history.keys():
                return
            
            # Change the flag of dict
            task_dict = self.tasks_history.pop(name)
            task_dict["flag"] = new_state
            self.tasks_history[name] = task_dict
            
            # Update button n if in all mode, auto deduction
            self.update_exec_button_n()
            
            # If reset mode, then reset all attrs in archive
            if new_state == -1:
                self.reset_an_archived_task(task_name = name)

            # Special for current mode
            # If sendtolist is True, then user is using current widget to modify the state by operations
            # Then, we need to inform the TaskList to change appearance
            if sendtolist == True:
                self.history_list.set_task_state_naive_noemit_(item, state=new_state)
            
            # Appearance is done in HistoryList class so no need to handle
            return
        
    # Rename the current task
    def rename_current_task(self, new_name:str):
        # Name strip cannot be empty
        new_name = new_name.strip()
        if not new_name:
            QMessageBox.warning(None, "错误", "任务名不可为空.")
            return

        # Handle name conflicts
        old_name = self.current_task_name
        new_name = new_name.strip()
        if self.current_task_name == new_name:
            return
        else:
            self.current_task_name = new_name
        while self.current_task_name in self.tasks_history.keys():
            self.current_task_name += "(1)" # avoid duplicating name
        
        # Pop out the old name in case it was saved once
        if old_name in self.tasks_history:
            _ = self.tasks_history.pop(old_name)

        # Insert name to Visable list
        item = self.history_list.currentItem()
        if item is not None:
            item.setText(self.current_task_name)
        return

    # Rename certain task
    def rename_certain_task(self, item, new_name:str):
        
        # Name strip cannot be empty
        new_name = new_name.strip()
        if not new_name:
            QMessageBox.warning(None, "错误", "任务名不可为空.")
            return

        # We ensure the name of item is still old name
        # If this item points to current, then call 
        if item.text() == self.current_task_name:
            return self.rename_current_task(new_name)
    
        # Modify something stored in history only
        old_name = item.text()
        new_name = new_name.strip()
        # Fail if the same are same or old name not in hist
        if old_name == new_name or old_name not in self.tasks_history.keys():
            return
        while new_name in self.tasks_history.keys():
            new_name += "(1)" # avoid duplicating name
        
        # Change the name of dict
        task_dict = self.tasks_history.pop(old_name)
        task_dict["name"] = new_name
        self.tasks_history[new_name] = task_dict
        
        # Insert name to Visable list
        item.setText(new_name)
        return

    # Reset the current Task
    def reset_current_task(self, *, new_id: bool=False, clear_stage_only: bool=False):
        """ Reset task will reset every attr and name, flag """

        # Reset name, flag; maintain id
        if new_id:
            self.cumulative_task_id += 1
        self.current_task_id = self.current_task_id if new_id == False else self.cumulative_task_id
        self.current_task_name = f"任务{self.current_task_id}"
        while self.current_task_name in self.tasks_history.keys():
            self.current_task_name += "(1)" # avoid duplicating name
        self.current_task_flag = -1 # Reset
        
        # Reset attributes
        self.current_image_path = None
        self.last_image_path = None
        self.current_output_text.clear()
        self.resolution_combo.setCurrentIndex(0)
        self.task_type_combo.setCurrentIndex(0)
        self.custom_prompt = ""
        self.current_progress_bar_val = 0

        # Reset Form
        self.clear_image(without_touch_state=clear_stage_only)
        self.clear_log(without_touch_state=clear_stage_only)
        self.progress_bar.setValue(0)
        self.user_prompt_ipt.setText("")

        # Now stage is cleared, then consider state
        
        # Reset current state and emit to reflect appearance
        if clear_stage_only == False:
            self.change_current_task_state(-1, sendtolist=True)
        
        return
    
    # Reset an archived task (adjust images and some attrs to init values)
    def reset_an_archived_task(self, task_name:str):
        
        # Try find the task
        task_name = task_name.strip()
        if self.tasks_history.get(task_name, None) is None:
            return # do nothing
        
        # Pop the dict
        task_dict = self.tasks_history.pop(task_name)
        
        # Modify flag, images, and other attrs
        if task_dict.get("flag", None) is not None:
            task_dict["flag"] = -1 # reset mode
        if task_dict.get("attr", None) is not None:
            task_dict_attr = task_dict.get("attr")
            if task_dict_attr.get("current_image_bytes", None):
                task_dict_attr["current_image_bytes"] = None
            if task_dict_attr.get("last_image_bytes", None):
                task_dict_attr["last_image_bytes"] = None
            if task_dict_attr.get("current_output_text", None):
                task_dict_attr["current_output_text"] = ""
            if task_dict_attr.get("resolution_combo_index", None):
                task_dict_attr["resolution_combo_index"] = 0
            if task_dict_attr.get("task_type_combo_index", None):
                task_dict_attr["task_type_combo_index"] = 0
            if task_dict_attr.get("custom_prompt", None):
                task_dict_attr["custom_prompt"] = ""
            if task_dict_attr.get("current_progress_bar_val", None):
                task_dict_attr["current_progress_bar_val"] = 0
            task_dict["attr"] = task_dict_attr
        
        # Insert it back
        self.tasks_history[task_name] = task_dict
        return
    
    # Archive the Current Task without resetting
    def archive_current_task(self):
        """ Archiving the current task meaning save imgs as bytes, and save other conditions and saves"""

        # Create a basic task dict
        task_dict = {
            "id": self.current_task_id,
            "name": self.current_task_name,
            "flag": self.current_task_flag,
            "attr": {
                # current_image_bytes
                # last_image_bytes
                # current_output_text
                # resolution_combo_index
                # task_type_combo_index
                # custom_prompt
                # current_progress_bar_val
            }
        }

        def _load_img_as_pil(path:str) -> Any | None:
            if os.path.exists(path):
                img = Image.open(path)
                return img
            else:
                return None

        # We assume name is unique since it's checked when setting a new name
        # Save images as bytes
        if self.current_image_path and os.path.exists(self.current_image_path):
            current_image_bytes = _load_img_as_pil(self.current_image_path)
            if current_image_bytes is not None:
                buffer = BytesIO()
                current_image_bytes.save(buffer, format="PNG")
                current_image_bytes = buffer.getvalue()
                if current_image_bytes:
                    task_dict["attr"]["current_image_bytes"] = current_image_bytes
        if self.last_image_path and os.path.exists(self.last_image_path):
            last_image_bytes = _load_img_as_pil(self.last_image_path)
            if last_image_bytes is not None:
                buffer = BytesIO()
                last_image_bytes.save(buffer, format="PNG")
                last_image_bytes = buffer.getvalue()
                if last_image_bytes:
                    task_dict["attr"]["last_image_bytes"] = last_image_bytes
        if task_dict["attr"].get("current_image_bytes", 1) == 1:
            task_dict["attr"]["current_image_bytes"] = None
        if task_dict["attr"].get("last_image_bytes", 1) == 1:
            task_dict["attr"]["last_image_bytes"] = None

        # Save text
        task_dict["attr"]["current_output_text"] = self.current_output_text.toPlainText()

        # Save combo values
        task_dict['attr']["resolution_combo_index"] = self.resolution_combo.currentIndex()
        task_dict['attr']["task_type_combo_index"] = self.task_type_combo.currentIndex()

        # Set remainings
        task_dict['attr']["custom_prompt"] = self.custom_prompt
        task_dict['attr']["current_progress_bar_val"] = self.current_progress_bar_val

        # Update History Dict
        self.tasks_history.update({self.current_task_name: task_dict})
        return
    
    # De-archive one task by name (in case sometimes it failed)
    def dearchive_one_task_by_name(self, taskname: str, *, silent: bool = True) -> Any:
        if self.tasks_history.get(taskname, None) is not None:
            # Got it and return it 
            return self.tasks_history.pop(taskname)
        else:
            # Not found
            if silent:
                return None
            else:
                raise KeyError(f"Invalid taskname {taskname} in dearchiving one task")

    # Create a New Task, Record the old to history, and Switch to new Task
    def create_new_task(self):
        """ A new task is created when user creates a task or at the beginning """

        # @TODO
        # currently not support switching when running an infr
        if self.model_isbusy == True:
            QMessageBox.warning(None, "错误", "模型正忙,请稍后执行操作")
            return

        # By default we archive the current and switch to new
        # Record Old
        try:
            self.archive_current_task()
        except:
            return # Errored

        # Create New with increment but without state updating (stage mode)
        self.reset_current_task(new_id=True, clear_stage_only=True)

        # Show in task bar
        self.history_list.addItem(self.current_task_name)
        
        # Auto focus on the new task
        self.history_list.setCurrentRow(self.history_list.count()-1)
        return
    
    # Set current task with elem in a task dict
    def _set_current_task_with_elem(self, task_dict:dict) -> bool:
        
        if task_dict is None:
            return False
        
        def _load_img_as_pil(path:str) -> Any | None:
            if os.path.exists(path):
                img = Image.open(path)
                if img:
                    return img
            else:
                return None
            
        def _save_img_as_file(img, *, path: str | None) -> bool:
            if img is None:
                return False
            img_bytes = img.tobytes()
            img_hash = hashlib.sha256(img_bytes).hexdigest()
            if path is None:
                # Auto Detect path
                path = os.path.join(self.current_wd, "tmp", f"{img_hash}.png")
            img.save(path, "PNG")
            return True

        def _write_bin_to_file(bin:bytes, path: str) -> None:
            if bin:
                try:
                    with open(path, "wb") as f:
                        f.write(bin)
                except:
                    pass
        
        # Restore images (faillable)
        if task_dict["attr"].get("current_image_bytes", None) is not None:
            fp = os.path.join(self.current_wd, "tmp", self._std_hash(task_dict["attr"]["current_image_bytes"]) + ".png")
            _write_bin_to_file(task_dict["attr"]["current_image_bytes"], fp)
            if os.path.exists(fp) == False:
                return False
            else:
                self.current_image_path = fp
        if task_dict["attr"].get("last_image_bytes", None) is not None:
            fp = os.path.join(self.current_wd, "tmp", self._std_hash(task_dict["attr"]["last_image_bytes"]) + ".png")
            _write_bin_to_file(task_dict["attr"]["last_image_bytes"], fp)
            if os.path.exists(fp) == False:
                return False
            else:
                self.last_image_path = fp

        # Restore current id, name, and flag
        if task_dict["id"] is not None:
            # Fuck Python. if 0: -> not enters
            # I just wanna check if it is None or not. Damn it
            self.current_task_id = task_dict["id"]
        if task_dict["name"] is not None:
            self.current_task_name = task_dict["name"]
        if task_dict["flag"] is not None:
            self.current_task_flag = task_dict["flag"]
        
        # Restore other attributes
        self.custom_prompt = task_dict["attr"]["custom_prompt"]

        # Reset Form
        if self.last_image_path and self.current_image_path:
            current_image_path = self.current_image_path # avoid being override
            self.load_image(self.last_image_path, without_touch_state=True)
            self.load_image(current_image_path, without_touch_state=True)
        elif self.current_image_path:
            self.load_image(self.current_image_path, without_touch_state=True)
        elif self.last_image_path:
            self.load_image(self.last_image_path, without_touch_state=True)
            self.clear_image(without_touch_state=True)
        else:
            # Neither True
            self.clear_image(without_touch_state=True)

        # The functionality above may change state, so we need to finally reset state to target type
        self.append_log(task_dict["attr"]["current_output_text"])
        self.resolution_combo.setCurrentIndex(task_dict["attr"]["resolution_combo_index"])
        self.task_type_combo.setCurrentIndex(task_dict["attr"]["task_type_combo_index"])
        self.progress_bar.setValue(task_dict["attr"]["current_progress_bar_val"])
        self.user_prompt_ipt.setText(task_dict["attr"]["custom_prompt"])
        
        # Reset state and emit into history list (well, to reflect the appearance)
        self.change_current_task_state(task_dict["flag"], sendtolist=True)
        # Note, since we reset the image and the state may be affected
        # Actually, we have fixed this by introducing without_touch_state 
        # But never mind, let's do again for demonstration
        
        return True
    
    # Switch to One Task
    def switch_to_task(self, item):

        # @TODO
        # currently not support switching when running an infr
        if self.model_isbusy == True:
            QMessageBox.warning(None, "错误", "模型正忙,请稍后执行操作")
            return

        # item: target
        # If target == current, done
        if self.current_task_name == item.text():
            return

        # Record Old
        try:
            self.archive_current_task()
        except:
            return # Errored
        
        # We need current name (before switching)
        current_name = self.current_task_name
    
        # Check item in the history_list
        if self.history_list.row(item) >= 0:

            # Reset current task
            self.reset_current_task(clear_stage_only=True)
            
            # Get the current task_dict
            task_dict = {
                "id": None,
                "name": None,
                "flag": None,
                "attr": {
                    # current_image_bytes
                    # last_image_bytes
                    # current_output_text
                    # resolution_combo_index
                    # task_type_combo_index
                    # custom_prompt
                    # current_progress_bar_val
                }
            }
            task_dict.update(self.tasks_history[item.text()])
        
            # If failed to load from elem, deregister current
            if self._set_current_task_with_elem(task_dict) == False:
                self.dearchive_one_task_by_name(current_name)
           
            return
            
    # Remove This Task (and align the other task id to make continuous) (focus on last one)
    def remove_current_task(self, *, sendtolist: bool = False):
        
        # If model is busy, deny
        if self.model_isbusy:
            QMessageBox.warning(None, "错误", "模型正忙时无法执行任务删除操作.")
            return
        
        # Archive current
        self.archive_current_task()

        # If item not in the dict, return
        if self.tasks_history.get(self.current_task_name, None) is None:
            return
        
        # If this is the only one, reject it
        if self.history_list.count() == 1:
            return
        
        # Okay. We do the following thing (aligns the QListWidget's definition)
        # > This is current object and have ensured to be inserted in the dict
        # 1. Get current item
        # 2. Pop it from the dict and get the no
        # 3. For items in the dict, if no > this_no, then - 1
        # 4. Clear current and load with previous one/next one (1st)
        item = self.history_list.findItems(self.current_task_name, Qt.MatchFlag.MatchExactly)[0]
        task_dict = self.tasks_history.pop(item.text())
        this_no = task_dict.get("id", None)
        if this_no is None:
            self.tasks_history[item.text()] = task_dict
            return # Rejected, abnormal, will not be here
        for k in list(self.tasks_history.keys()):
            d_ref = self.tasks_history.pop(k)
            d_ref_id = d_ref.get("id", -1)
            if d_ref_id > this_no:
                d_ref["id"] = d_ref_id - 1
            self.tasks_history[k] = d_ref       
        self.reset_current_task(clear_stage_only=True) # We will fill this with last (or next if the 1st one)
        id_to_set = -1
        name_to_set = ""
        if this_no == 0:
            # Next one
            for k in self.tasks_history.keys():
                d_ref = self.tasks_history[k]
                d_ref_id = d_ref.get("id", -1)
                if d_ref_id == this_no:
                    id_to_set = d_ref_id
                    name_to_set = d_ref.get("name", "")
                    break
        else:
            # Last one
            for k in self.tasks_history.keys():
                d_ref = self.tasks_history[k]
                d_ref_id = d_ref.get("id", -1)
                if d_ref_id == this_no - 1:
                    id_to_set = d_ref_id
                    name_to_set = d_ref.get("name", "")
                    break
        self._set_current_task_with_elem(self.tasks_history[name_to_set])

        # Cumulative -= 1
        self.cumulative_task_id -= 1

        # Take the item from History List
        if sendtolist:
            if item is not None:
                self.history_list._try_take_item(item)
                # If first focus on next; else focus on last
                self.history_list.setCurrentItem(self.history_list.item(id_to_set))

        return
    
    # Remove Certain Task (and align the other task id to make continuous) (focus unchanged)
    def remove_certain_task(self, item, *, sendtolist: bool = False):
        # BTW, for this function, sendtolist may be True for most cases

        # If model is busy, deny
        if self.model_isbusy:
            QMessageBox.warning(None, "错误", "模型正忙时无法执行任务删除操作.")
            return

        # We ensure the name of item is still old name
        # If this item points to current, then call 
        if item.text() == self.current_task_name:
            return self.remove_current_task(sendtolist=sendtolist)

        # If this is the only one and not current, reject it
        if self.history_list.count() == 1:
            return
        
        # If item not in the dict, return
        if self.tasks_history.get(item.text(), None) is None:
            return
        
        # Okay. We do the following thing (aligns the QListWidget's definition)
        # > This is not current object, so do what?
        # 1. Archive current object to make it in the dict
        # 2. Pop it from the dict and get the no
        # 3. For items in the dict, if no > this_no, then - 1
        # 4. For current focus item, if no > this_no, then -1
        self.archive_current_task()
        task_dict = self.tasks_history.pop(item.text())
        this_no = task_dict.get("id", None)
        if this_no is None:
            self.tasks_history[item.text()] = task_dict
            return # Rejected, abnormal, will not be here
        for k in list(self.tasks_history.keys()):
            d_ref = self.tasks_history.pop(k)
            d_ref_id = d_ref.get("id", -1)
            if d_ref_id > this_no:
                d_ref["id"] = d_ref_id - 1
            self.tasks_history[k] = d_ref
        if self.current_task_id > this_no:
            self.current_task_id -= 1

        # Cumulative -= 1
        self.cumulative_task_id -= 1
        
        # Take the item from History List
        if sendtolist:
            self.history_list._try_take_item(item)
        return

    # Remove All Tasks
    def remove_all_tasks(self, *, reinit_0: bool = True):

        # If model is busy, deny
        if self.model_isbusy:
            QMessageBox.warning(None, "错误", "模型正忙时无法执行任务删除操作.")
            return
        
        # Clear History dict
        self.tasks_history.clear()
        
        # Clear visable list 
        self.history_list.clear()
        
        # Reiniitalize if required
        if reinit_0 == True:
            
            # Get everything to init
            self.current_task_id = 0       # a valid task starts from 0
            self.cumulative_task_id = 0    # by default, the 1st task
            self.current_task_name = "任务0"
            self.current_task_flag = -1    # Task flag: -1 for notready, 0 for ready but unfinished, 1 for completed
            
            # Set current task to visable table
            self.history_list.addItem(self.current_task_name)
            self.history_list.setCurrentItem(self.history_list.item(0))  # Select the current one
        
            # Reset current task
            self.reset_current_task()
            # This must be after reinit the history list.
            # Or there will be no element and throw an error
           
        else:
            self.current_task_id = 0       # a valid task starts from 0
            self.cumulative_task_id = 0    # by default, the 1st task

        return
    
    # Save Tasks 
    def save_task_img(self):

        # If model is busy, deny
        if self.model_isbusy:
            QMessageBox.warning(None, "错误", "模型正忙时无法执行镜像保存操作.")
            return
        
        # Request a place
        file_path = ""
        folder_path = QFileDialog.getExistingDirectory(self, "选择镜像保存目录", self.last_task_img_folder if self.last_task_img_folder else self.current_wd)
        if not folder_path:
            return
        if folder_path and os.path.exists(folder_path):
            file_path = os.path.join(folder_path, "task_image_"+datetime.now().strftime("%Y-%m-%d %H-%M-%S")+".img")
        else:
            QMessageBox.warning(None, "错误", "无效的存储目录.")
            return
        
        # Archive the current task
        self.archive_current_task()
        
        # Save the dict as a pickle
        tasks = {"attr": "DSOCR.image", "version": DSOCR_UI_DEFAULT_KRLVER, "data": deepcopy(self.tasks_history)}
        save(tasks, file_path, kompress=zstandard, protocol=5)

        # Got you! Update the last img folder
        self.last_task_img_folder = folder_path
        
        QMessageBox.information(None, "成功", f"成功保存镜像于{file_path}")
    
    # Reload Back Tasks
    def reload_task_img(self, *, given_a_task_dict: dict | None = None):
        
        # If model is busy, deny
        if self.model_isbusy:
            QMessageBox.warning(None, "错误", "模型正忙时无法执行镜像加载操作.")
            return

        # task img buffer
        taskimg = None

        # If given, directly evaluate
        if given_a_task_dict is not None and len(given_a_task_dict) > 0:
            taskimg = deepcopy(given_a_task_dict)
        
        # Else, get from disk
        else:
            # Request a file
            file_path, _ = QFileDialog.getOpenFileName(self, "选择需要加载的镜像", "", "DSOCR.image (*.img)")
            if not file_path:
                return
            try:
                if file_path and os.path.exists(file_path):
                    taskimg = load(file_path, kompress=zstandard)
                else:
                    raise RuntimeError("Bad filepath given")
                if taskimg is None:
                    raise RuntimeError("Failed to load the task image")
                if isinstance(taskimg, dict) == False:
                    raise TypeError("Image is not a dict")
                # @TODO add version check logic in the future
                if taskimg.get("attr", None) != "DSOCR.image" or taskimg.get("data", None) is None:
                    raise ValueError("Bad Image or already corrupted")
                taskimg = taskimg["data"]
            except:
                QMessageBox.warning(None, "错误", "无效或损坏的镜像.")
                return
            
        # Remove all tasks
        self.remove_all_tasks(reinit_0=False) # without even reinitialize the 0
        
        # Manually refresh current but I know we have no history list 
        # so there will be an error. But we catch it
        try:
            # Without emiting history list <\solved>
            self.reset_current_task(new_id=False, clear_stage_only=True)
        except:
            pass
        
        # Load the new task and switch to task 0
        self.tasks_history = taskimg

        # Therefore, another global variable is needed: cumulative counter
        self.cumulative_task_id = len(self.tasks_history) - 1
        
        # Set visable list 
        # Must be first since we need to ensure enough elements are loaded
        td = {"id": 99999999999999,} # placeholder, item having minimum id
        idls = {"id":[], "name":[], "flag":[]}
        for k in self.tasks_history.keys():
            if self.tasks_history[k].get("id", 99999999999999) < td.get("id"):
                td = self.tasks_history[k]
            if self.tasks_history[k].get("id") is not None and self.tasks_history[k].get("name") is not None and self.tasks_history[k].get("flag") is not None:
                idls["id"].append(self.tasks_history[k].get("id"))
                idls["name"].append(self.tasks_history[k].get("name"))
                idls["flag"].append(self.tasks_history[k].get("flag"))
        pdf = pandas.DataFrame(idls).sort_values(by="id")
        for i in range(len(pdf)):
            # Using pandas. Must cast type or may use numpy types
            self.history_list.addItem(str(pdf["name"].iloc[i]), state=int(pdf["flag"].iloc[i]))
        self.history_list.setCurrentItem(self.history_list.item(0)) # Like init
            
        # Load the 1st index task from elem (backend database reset)
        self._set_current_task_with_elem(deepcopy(td)) # recommend to deepcopy
        
        return
        
    # Create a streamer for online inference task
    def create_proper_streamer(self, logging=True, *args, **kwargs):
        
        # @TODO
        # Need to support non-current task id
        thread_task_id = self.current_task_id
        thread_self = self.th_inference_task_thread

        # New Special Class designed for log processing
        class TextAccumulatorLog(TextAccumulator):
            
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

                # Record the main thread object
                self.thread_self = thread_self

                # Record the task id for reference
                self.thread_task_id = thread_task_id
                
            def on_finalized_text(self, text, stream_end=False):
                # Process to drop "<｜end▁of▁sentence｜>" if any
                text = text.replace(r"<｜end▁of▁sentence｜>", "")
                
                # Append to log if on focus
                self.thread_self.log_appended.emit(text)
                if stream_end == True:
                    self.thread_self.log_appended.emit("\n")

                # Append to task history if off focus

                # Need synchron to avoid saving and switching collisions
                # @TODO

                super().on_finalized_text(text=text, stream_end=stream_end)
                
        # If use logging then instantiate the Log instance
        if logging:
            return TextAccumulatorLog(*args, skip_special_tokens=False, **kwargs)
        else:
            # STRONGLY NOT recommended
            return TextAccumulator(*args, skip_special_tokens=False, **kwargs)
    # @TODO
    # for MT support, we need not only to bind this to current output 
    # and check if current is on focus then output to text bar
    # else only save this to history output buffer
    #
    # May need more synchron 

    # Execute helpers: initialize model runtime variables to indicate busy
    def _execute_on_task_runtime_var_init(self, on_task_names: list):
        if self.model_isbusy == True:
            # WTF! Impossible. Bug it
            raise RuntimeError("Attempt to take over model internally when model is busy/ Bug")
        if on_task_names:
            self.model_isbusy = True
            self.model_running_on_task_name = on_task_names[0]
            self.model_assigned_task_list_ofnames = on_task_names
            self.update_title()

    # Execute helpers: switch running task name context (on another task now)
    def _execute_on_task_runtime_adv_step(self, last_task_name: str, to_task_name: str | None):
        if self.model_isbusy == False:
            # WTF! Impossible. Bug it
            raise RuntimeError("Attempt to advance task context when model is not running")
        # Last must be good, else return
        if last_task_name is None:
            return
        # We will set last name to completed (1)
        items: list = self.history_list.findItems(last_task_name, Qt.MatchFlag.MatchExactly)
        item = items[0] if items else None
        if not item or not item.text():
            return
        self.change_certain_task_state(item, 1, sendtolist=True) # actively make the change
                       
        # And move self.model_running_on_task_name to next task name
        if to_task_name:
            # In case the last one
            self.model_running_on_task_name = to_task_name
        else:
            self.model_running_on_task_name = None
        
    # Execute helpers: finally, set model to not busy when all done
    def _execute_on_task_runtime_all_done(self):
        if self.model_isbusy == False:
            # WTF! Impossible. Bug it
            raise RuntimeError("Attempt to put back model internally when model is already idle/ Bug")
        self.model_isbusy = False
        self.model_running_on_task_name = None
        self.model_assigned_task_list_ofnames = []
        self.update_title()

    # Execute outer helper: is assigned task including a certain task
    def execute_helper_is_assigned_with(self, task_name:str) -> bool:
        # Who calls this?
        # Well, when modelling is running. The assigned tasks are not able to make any WRITE modification
        # But for other tasks, you are allowed to do anything
        # Moreover, you are not allowed to delete all/ delete any when the model is running
        # or it will break the id sequence since we always ensure ids are from 0 to n continguously
        # So, WRITE operations will always check this to make sure it is legit
        if self.model_isbusy == False:
            return False
        if not self.model_assigned_task_list_ofnames:
            return False
        if task_name in self.model_assigned_task_list_ofnames:
            return True
        else:
            return False
        
    # Execute outer helper: is running on a certain task
    def execute_helper_is_running_on(self, task_name:str) -> bool:
        # Same idea as above
        if self.model_isbusy == False:
            return False
        if task_name == self.model_running_on_task_name:
            return True
        else:
            return False

    # Execute the current OCR Task
    def execute_current_task(self):

        # If model is busy, then wrong
        if self.model_isbusy == True:
            QMessageBox.warning(None, "错误", "模型正忙,请稍后执行推理")
            return
    
        # No iamge, then wrong
        if not hasattr(self, 'current_image_path') or not self.current_image_path:
            QMessageBox.warning(None, "错误", "请至少上传一张图片以执行推理")
            return
        
        # Flag is not 0, then wrong
        if self.current_task_flag != 0:
            QMessageBox.warning(None, "错误", "任务状态必须为<未处理>以执行推理")
            return
        
        # Clear the Output Text Box
        self.clear_log(without_touch_state=True, granted_in_model_running=True) # No impact on state, not invoked by user
        # Grant the permission in model running
        
        # Set model is busy and runtime attributes
        self._execute_on_task_runtime_var_init([self.current_task_name])
        # On and assigned with current task
        
        # Save current resolution type and task type
        self.current_resolution_type = self.resolution_combo.currentText()
        self.current_task_type = self.task_type_combo.currentText()

        # Archive the current version (after clearing log) to history list 
        # in case change of focus
        self.archive_current_task()

        # Get current task name 
        model_running_on_current_task_name = self.current_task_name
        
        # API Call Function Placeholder
        # User inserts logic here, e.g.:
        # def api_call(resolution, task_type, image_path):
        #     # Call external async program, return success
        #     pass
        def _api_call():
            if not DSOCR_DEBUG:
                
                # Infer output path
                # We use a lazy mode: hash the filepath
                output_folder = os.path.join(self.current_wd, "o", self._std_hash(self.current_image_path))
                os.makedirs(output_folder, exist_ok=True)
                
                # Call inference by custom Inference API
                try:
                    res, streamer = self.model.infer(
                        prompt=self._task_mapping(self.current_task_type.strip(), inputs=self.custom_prompt.strip()),
                        image_file=self.current_image_path,
                        output_path=output_folder,
                        template=self.current_resolution_type.strip(), 
                        save_results=True,
                        test_compress=False,
                        eval_mode=False,
                        streamer=self.create_proper_streamer(True, tokenizer=self.model.tokenizer),
                        font_path=DSOCR_UI_DEFAULT_FTPATH
                    )
                except:
                    return False
                    
                # res: str, str, pil, None
                oresl, resl, pil, _ = res
                
                # Reset the image to processed image
                self.load_image(os.path.join(output_folder, "result_with_boxes.jpg"), without_touch_state=True)
                # Do not refresh state. We will set completed as below

            # Advance one step
            self._execute_on_task_runtime_adv_step(last_task_name=model_running_on_current_task_name, to_task_name=None)
            # Last one, no advancement anymore

            # All Done will be set in on_task_completed (see below)
                 
            return True  # or False on error
        
        # Start async thread
        self.th_inference_task_thread = AsyncTaskExecutor(_api_call)
        self.th_inference_task_thread.progress_updated.connect(self.progress_bar.setValue)
        self.th_inference_task_thread.log_appended.connect(self.append_log)
        self.th_inference_task_thread.task_completed.connect(self.on_task_completed)
        self.th_inference_task_thread.start()
    
    # Execute all Selected Tasks
    def execute_all_selected_tasks(self):

        # Collect ready tasks
        n_ready_tasks = self.count_ready_tasks()

        # If no ready tasks, raise Error
        if n_ready_tasks == 0:
            QMessageBox.warning(None, "错误", "当前没有任务状态为<未处理>的任务，无法执行推理")
            return

        #Not implemented
        # @TODO
        #NOW, JUST CALL CURRENT
        QMessageBox.warning(None, "错误", "还没有实现多任务处理呢!")
        # self.execute_current_task()

    # @TODO

    # When Task Finished
    def on_task_completed(self, success):
        
        if success:
            pass

        # Model is All Done
        self._execute_on_task_runtime_all_done()


# Command Line Server (Wrapper)
class ImageProcessingServer:
    
    pass
    # @TODO

# Entry Point
def main(isgui: bool=True, app_kwargs={}, gui_kwargs={}, **kwargs):
    
    # Start GUI Application
    if isgui:
        if 1:
        # try:
            app = QApplication(sys.argv)
            # Appilication Id
            appid = 'pyqt6.python.ui.DSOCR'
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(appid)
            # Application Font
            appfont = QFont(app_kwargs.get("font", "宋体"), app_kwargs.get("font_size", 10)) 
            app.setFont(appfont)
            # Initialize Main Window
            window = ImageProcessingGUI(**gui_kwargs)
            window.setWindowIcon(QIcon(DSOCR_UI_DEFAULT_ICPATH))
            window.show()
            sys.exit(app.exec())
        # except Exception as e:
        #     print(e)

    # Start server
    else:
        pass

if __name__ == "__main__":
    
    # Cast Nonetype 
    def _handle_Nonetype(dict_) -> dict:
        ks = dict_.keys()
        for k in ks:
            if dict_.get(k, None) == "None":
                dict_[k] = None
        return dict_
    
    # Formal check and handling the args
    def _handle_validity(dict_) -> dict:
        dict_ = _handle_Nonetype(dict_)
        # Working Directory
        if dict_.get("wd", None) is not None:
            v = dict_["wd"].strip()
            if os.path.exists(v) == False:
                try:
                    os.makedirs(v, exist_ok=False)
                except:
                    # Even fail to make one, then back to None
                    dict_["wd"] = None
        # Theme (Will Replace by lower if exists)
        if dict_.get("theme", None) is not None:
            v = dict_["theme"].lower()
            if v not in ("light", "dark"):
                dict_["theme"] = None
            else:
                dict_["theme"] = v
        # Inference Device, cpu, cuda, ...
        if dict_.get("inference_device", None) is not None:
            v = dict_["inference_device"].lower().strip()
            if v != 'cpu' and v.startswith("cuda") == False:
                dict_["inference_device"] = None
        # Quantization Mode
        if dict_.get("quant", None) is not None:
            v = dict_["quant"].lower().strip()
            if v not in ("nf4", "bnb_int8", "bfloat16", "float16", "float32"):
                dict_["quant"] = None
        # Magic  
        if dict_.get("enable_magic", None) is not None:
            v = dict_["enable_magic"].lower().strip()
            if v not in ("all", "gal"):
                dict_["enable_magic"] = None
        return dict_
                
    # Parse config json
    if os.path.exists("./config.json"):
    
        # Parse json
        config = ""
        with open("config.json", "r") as f:
            config = json.load(f)
        
        if config['__attr__'] == 'DSOCR':
            appconfig = _handle_Nonetype(config["__config__"]["app"])
            guiconfig = _handle_validity(config["__config__"]['ImageProcessingGUI'])
            main(True, app_kwargs=appconfig, gui_kwargs=guiconfig)
    