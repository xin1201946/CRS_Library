import threading
import queue
import time
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text

class ServerGUI:
    def __init__(self, server_url, use_https, ssh_path, AdvanceAPISetting, logSwitch, logs=None, client_func=None):
        """
        初始化 ServerGUI 类的实例。
        Initialize an instance of the ServerGUI class.

        :param server_url: 服务器监听地址
        :param server_url: Server listening address
        :param use_https: 是否使用 HTTPS 服务
        :param use_https: Whether to use HTTPS service
        :param ssh_path: SSH 路径
        :param ssh_path: SSH path
        :param AdvanceAPISetting: 是否启用高级 API 设置
        :param AdvanceAPISetting: Whether to enable advanced API settings
        :param logSwitch: 是否允许前端访问日志
        :param logSwitch: Whether to allow front-end access to logs
        :param logs: 日志列表，默认为空
        :param logs: List of logs, default is empty
        :param client_func: 用于获取客户端信息的函数，默认为 None
        :param client_func: Function to get client information, default is None
        """
        self.server_url = server_url
        self.use_https = use_https
        self.ssh_path = ssh_path
        self.AdvanceAPISetting = AdvanceAPISetting
        self.logSwitch = logSwitch
        self.logs = logs or []

        self.clients = []
        self.console = Console()
        self.layout = Layout()

        self.sysinfo = []

        self.queue = queue.Queue()
        self.running = threading.Event()
        self.running.set()

        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
        )
        self.layout["body"].split_row(
            Layout(name="left", size=40),
            Layout(name="right"),
        )

        self.clients_lock = threading.Lock()
        self.logs_lock = threading.Lock()

        self.client_func = client_func

    def showGUI(self):
        """
        显示图形用户界面（GUI），并持续更新界面内容。
        Display the graphical user interface (GUI) and continuously update the interface content.
        """
        self.layout["header"].update(Panel(Text("CCRS", style="bold white", justify='center')))
        self.layout["left"].split_column(
            Layout(name="config", size=10),
            Layout(name="table"),
        )

        self.log_panel = Panel("", title="Logging area", border_style="red")
        self.layout["right"].update(self.log_panel)

        with Live(self.layout, refresh_per_second=4, screen=True) as live:
            while self.running.is_set():
                self.process_queue()
                self.refresh_GUI()
                live.refresh()
                time.sleep(0.25)  # 控制刷新频率
                # Control the refresh frequency

    def process_queue(self):
        """
        处理队列中的消息，根据消息类型执行相应操作。
        Process messages in the queue and perform corresponding operations based on the message type.
        """
        while not self.queue.empty():
            try:
                message = self.queue.get_nowait()
                if message["event"] == "New device":
                    self.add_client(message["UUID"], message["aID"])
                elif message["event"] == "Log event":
                    self.log_event(message)
            except queue.Empty:
                break

    def refresh_GUI(self):
        """
        刷新图形用户界面（GUI）的各个部分，包括配置信息、客户端列表、服务器信息和日志显示。
        Refresh various parts of the graphical user interface (GUI), including configuration information,
        client list, server information, and log display.
        """
        if self.client_func is not None:
            with self.clients_lock:
                uuid_aid_dict = self.client_func()
                self.clients = [[uuid, aid] for uuid, aid in uuid_aid_dict.items()]
        config_text = f"""
[b]Basic Config[/b]
- Server listening address: {self.server_url}
- HTTPS Service: {'[red]OFF[/red]' if not self.use_https else '[green]ON[/green]'}
- SSH PATH: {self.ssh_path}
- Advanced API Settings: {'[red]OFF[/red]' if not self.AdvanceAPISetting else '[green]ON[/green]'}
- Allow front-end access to logs: {'[green]ON[/green]' if self.logSwitch else '[red]OFF[/red]'}
"""
        config_panel = Panel(config_text, title="Basic Config", border_style="blue")
        self.layout["config"].update(config_panel)

        table1 = Table(title="Registered Clients", title_style="bold cyan")
        table1.add_column("UUID", style="cyan", justify="center")
        table1.add_column("aID", style="magenta", justify="center")
        with self.clients_lock:
            for client in self.clients:
                table1.add_row(client[0], client[1])

        table2 = Table(title="Server Info", title_style="bold cyan")
        table2.add_column("CPU", style="cyan", justify="center")
        table2.add_column("RAM", style="magenta", justify="center")
        with self.clients_lock:
            for client in self.sysinfo:
                table2.add_row(client[0], client[1])

        self.layout["table"].update(Panel(table1, title="Information area", border_style="green"))

        with self.logs_lock:
            log_text = "\n".join(self.logs[-20:])  # 获取最新的20条日志
            # Get the latest 20 logs
        log_panel = Panel(log_text, title="Log display area", border_style="red")
        self.layout["right"].update(log_panel)

    def add_client(self, UUID, aID):
        """
        向客户端列表中添加一个新的客户端。
        Add a new client to the client list.

        :param UUID: 客户端的唯一标识符
        :param UUID: Unique identifier of the client
        :param aID: 客户端的另一个标识符
        :param aID: Another identifier of the client
        """
        with self.clients_lock:
            self.clients.append([UUID, aID])

    def log_event(self, log_data):
        """
        记录一个事件到日志列表中。
        Record an event to the log list.

        :param log_data: 包含事件信息的字典，应包含时间戳、事件、结果和备注
        :param log_data: Dictionary containing event information, should include timestamp, event, result, and remark
        """
        with self.logs_lock:
            self.logs.append(f"{log_data['timestamp']} - {log_data['event']} - {log_data['result']} - {log_data['remark']}")

    def stop(self):
        """
        停止图形用户界面（GUI）的更新循环。
        Stop the update loop of the graphical user interface (GUI).
        """
        self.running.clear()