"""
历史记录查看对话框
以表格形式展示检测历史，并显示基本统计信息。
"""
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QTabWidget, QWidget,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

import history


DIALOG_STYLE = """
QDialog { background-color: #1e1e2e; }
QWidget { background-color: #1e1e2e; color: #cdd6f4;
          font-family: "Segoe UI","Microsoft YaHei",sans-serif; font-size: 13px; }
QGroupBox { border: 1px solid #45475a; border-radius: 8px; margin-top: 10px;
            padding: 12px 8px 8px 8px; font-weight: bold; color: #89b4fa; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; }
QPushButton { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a;
              border-radius: 6px; padding: 6px 14px; min-height: 28px; }
QPushButton:hover { background-color: #45475a; border-color: #89b4fa; }
QPushButton#btnDanger { background-color: #f38ba8; color: #1e1e2e; border: none; font-weight: bold; }
QPushButton#btnDanger:hover { background-color: #eba0ac; }
QTableWidget { background-color: #11111b; color: #cdd6f4; border: 1px solid #45475a;
               border-radius: 6px; gridline-color: #313244;
               font-family: "Consolas","Cascadia Code",monospace; font-size: 12px; }
QTableWidget::item { padding: 4px; }
QTableWidget::item:selected { background-color: #45475a; }
QHeaderView::section { background-color: #313244; color: #89b4fa; padding: 6px;
                       border: 1px solid #45475a; font-weight: bold; }
QLabel#statLabel { color: #f9e2af; font-size: 14px; font-weight: bold; }
"""


class HistoryDialog(QDialog):
    """历史记录查看对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("检测历史记录")
        self.setGeometry(100, 80, 900, 600)
        self.setStyleSheet(DIALOG_STYLE)
        self._init_ui()
        self._load_data()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        tabs = QTabWidget()

        # ---- Tab 1: 统计概览 ----
        tab_stats = QWidget()
        stats_layout = QVBoxLayout(tab_stats)

        self.stat_group = QGroupBox("统计概览")
        stat_inner = QVBoxLayout(self.stat_group)

        self.lbl_total = QLabel("总检测次数: --")
        self.lbl_total.setObjectName("statLabel")
        self.lbl_objects = QLabel("累计检测目标: --")
        self.lbl_objects.setObjectName("statLabel")
        self.lbl_avg_latency = QLabel("平均延迟: -- ms")
        self.lbl_avg_latency.setObjectName("statLabel")
        self.lbl_time_range = QLabel("时间范围: --")
        self.lbl_time_range.setObjectName("statLabel")

        stat_inner.addWidget(self.lbl_total)
        stat_inner.addWidget(self.lbl_objects)
        stat_inner.addWidget(self.lbl_avg_latency)
        stat_inner.addWidget(self.lbl_time_range)
        stat_inner.addStretch()

        stats_layout.addWidget(self.stat_group)
        tabs.addTab(tab_stats, "统计")

        # ---- Tab 2: 记录表格 ----
        tab_records = QWidget()
        records_layout = QVBoxLayout(tab_records)

        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "时间", "模型", "任务", "源", "源类型", "目标数", "延迟(ms)"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)

        records_layout.addWidget(self.table)
        tabs.addTab(tab_records, "记录列表")

        # ---- Tab styling ----
        tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #45475a; border-radius: 6px; background: #1e1e2e; }
            QTabBar::tab { background: #313244; color: #cdd6f4; padding: 8px 16px;
                          border: 1px solid #45475a; border-bottom: none;
                          border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 2px; }
            QTabBar::tab:selected { background: #1e1e2e; border-bottom: 2px solid #89b4fa; color: #89b4fa; }
            QTabBar::tab:hover { background: #45475a; }
        """)
        layout.addWidget(tabs)

        # ---- 按钮 ----
        btn_layout = QHBoxLayout()
        self.btn_refresh = QPushButton("刷新")
        self.btn_clear = QPushButton("清空历史")
        self.btn_clear.setObjectName("btnDanger")
        self.btn_close = QPushButton("关闭")

        btn_layout.addWidget(self.btn_refresh)
        btn_layout.addWidget(self.btn_clear)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_close)
        layout.addLayout(btn_layout)

        self.btn_refresh.clicked.connect(self._load_data)
        self.btn_clear.clicked.connect(self._clear_history)
        self.btn_close.clicked.connect(self.close)

    def _load_data(self):
        """加载/刷新数据"""
        records = history.get_records()
        stats = history.get_statistics()

        # 更新统计标签
        total = stats.get("total", 0)
        self.lbl_total.setText(f"总检测次数: {total}")
        self.lbl_objects.setText(f"累计检测目标: {stats.get('total_objects', 0)}")
        self.lbl_avg_latency.setText(f"平均延迟: {stats.get('avg_latency_ms', 0):.1f} ms")
        if total > 0:
            self.lbl_time_range.setText(
                f"时间范围: {stats.get('first_record', '')} ~ {stats.get('last_record', '')}"
            )
        else:
            self.lbl_time_range.setText("时间范围: --")

        # 填充表格（最新在前）
        self.table.setRowCount(0)
        for r in reversed(records):
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(r.get("timestamp", "")))
            self.table.setItem(row, 1, QTableWidgetItem(r.get("model", "")))
            self.table.setItem(row, 2, QTableWidgetItem(r.get("task", "")))
            self.table.setItem(row, 3, QTableWidgetItem(r.get("source", "")))
            self.table.setItem(row, 4, QTableWidgetItem(r.get("source_type", "")))
            self.table.setItem(row, 5, QTableWidgetItem(str(r.get("obj_count", 0))))
            self.table.setItem(row, 6, QTableWidgetItem(str(r.get("latency_ms", 0))))

    def _clear_history(self):
        reply = QMessageBox.question(
            self, "确认", "确定要清空所有历史记录吗？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            history.clear_history()
            self._load_data()
