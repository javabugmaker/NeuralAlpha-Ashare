from __future__ import annotations

import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk


class ResearchApp:
    def __init__(self, root: tk.Tk, config_path: Path) -> None:
        self.root = root
        self.config_path = config_path.resolve()
        self.messages: queue.Queue[str] = queue.Queue()
        self.running = False
        root.title("NeuralAlpha A股研究台")
        root.geometry("900x620")
        root.minsize(720, 480)
        shell = ttk.Frame(root, padding=18)
        shell.pack(fill="both", expand=True)
        ttk.Label(shell, text="NeuralAlpha A股研究台", font=("Segoe UI", 22, "bold")).pack(
            anchor="w"
        )
        ttk.Label(shell, text="LightGBM · 严格时点 · t+1 成交 · GUI 与 CLI 同源").pack(
            anchor="w", pady=(2, 16)
        )
        buttons = ttk.Frame(shell)
        buttons.pack(fill="x")
        for label, command in (
            ("环境检查", "doctor"),
            ("更新行情", "update"),
            ("构建数据", "build"),
            ("训练模型", "train"),
            ("滚动验证", "walk-forward"),
            ("组合回测", "backtest"),
            ("生成日报", "daily"),
            ("生成周报", "weekly"),
        ):
            ttk.Button(buttons, text=label, command=lambda item=command: self.run(item)).pack(
                side="left", padx=(0, 8), pady=(0, 12)
            )
        self.status = tk.StringVar(value="就绪")
        ttk.Label(shell, textvariable=self.status).pack(anchor="w")
        self.log = tk.Text(shell, wrap="word", font=("Consolas", 10), bg="#0b1118", fg="#dce8f5")
        self.log.pack(fill="both", expand=True, pady=(8, 0))
        self.root.after(100, self._drain)

    def run(self, command: str) -> None:
        if self.running:
            messagebox.showinfo("任务运行中", "请等待当前任务结束。")
            return
        self.running = True
        self.status.set(f"正在运行：{command}")
        threading.Thread(target=self._worker, args=(command,), daemon=True).start()

    def _worker(self, command: str) -> None:
        argv = [
            sys.executable,
            "-m",
            "neural_alpha_ashare.cli",
            "--config",
            str(self.config_path),
            command,
        ]
        process = subprocess.Popen(
            argv,
            cwd=self.config_path.parent.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            self.messages.put(line)
        return_code = process.wait()
        self.messages.put(f"\n任务结束，退出码 {return_code}\n")
        self.messages.put("__DONE__")

    def _drain(self) -> None:
        try:
            while True:
                message = self.messages.get_nowait()
                if message == "__DONE__":
                    self.running = False
                    self.status.set("就绪")
                else:
                    self.log.insert("end", message)
                    self.log.see("end")
        except queue.Empty:
            pass
        self.root.after(100, self._drain)


def launch_gui(config_path: str | Path = "config/default.yaml") -> None:
    root = tk.Tk()
    ResearchApp(root, Path(config_path))
    root.mainloop()
