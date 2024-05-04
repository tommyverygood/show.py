import random, glob, datetime, sys, itertools, math, time
from collections import defaultdict
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout,
                             QWidget, QTextEdit, QLineEdit, QComboBox, QHBoxLayout,
                             QScrollArea, QTextBrowser, QStackedWidget, QLabel, QFileDialog)

from PyQt5.QtCore import pyqtSignal, QThread, Qt

def generate_combinations(elements, k):
    return list(itertools.combinations(elements, k))

def subset_cover_graph(n_numbers, k, j, s):
    j_subsets = generate_combinations(n_numbers, j)
    k_combinations = generate_combinations(n_numbers, k)
    cover_graph = defaultdict(list)
    for k_comb in k_combinations:
        for j_sub in j_subsets:
            if any(set(subset).issubset(k_comb) for subset in generate_combinations(j_sub, s)):
                cover_graph[tuple(k_comb)].append(tuple(j_sub))
    return cover_graph

def all_j_subsets_covered(cover_graph, solution):
    all_j_subsets = set(itertools.chain(*cover_graph.values()))
    covered_j_subsets = set(itertools.chain(*[cover_graph[k] for k in solution]))
    return covered_j_subsets == all_j_subsets

def simulated_annealing(cover_graph, n_numbers, T=10000, T_min=0.001, alpha=0.99, time_limit=8):
    start_time = time.time()
    k_combinations = list(cover_graph.keys())
    random.shuffle(k_combinations)
    current_solution = k_combinations[:len(k_combinations)//4]
    while not all_j_subsets_covered(cover_graph, current_solution):
        current_solution.append(random.choice([k for k in k_combinations if k not in current_solution]))
    current_energy = len(current_solution)
    while T > T_min:
        if time.time() - start_time > time_limit:
            print("Switching to greedy algorithm due to time limit.")
            return greedy_set_cover(cover_graph)
        for _ in range(50):
            new_solution = current_solution[:]
            if random.random() > 0.5 and len(new_solution) > 1:
                new_solution.remove(random.choice(new_solution))
            else:
                possible_additions = [k for k in k_combinations if k not in new_solution]
                if possible_additions:
                    new_solution.append(random.choice(possible_additions))
            if all_j_subsets_covered(cover_graph, new_solution):
                new_energy = len(new_solution)
                if new_energy < current_energy or math.exp((current_energy - new_energy) / T) > random.random():
                    current_solution = new_solution
                    current_energy = new_energy
        T *= alpha
    return current_solution

def greedy_set_cover(cover_graph):
    covered = set()
    selected_k_combs = []
    while any(j_sub not in covered for j_subsets in cover_graph.values() for j_sub in j_subsets):
        best_k_comb = max(cover_graph, key=lambda k: len(set(cover_graph[k]) - covered))
        selected_k_combs.append(best_k_comb)
        covered.update(cover_graph[best_k_comb])
    return selected_k_combs

def save_to_database(m, n, k, j, s, results):
    filename = f"{m}_{n}_{k}_{j}_{s}.txt"  # 直接使用参数值创建文件名
    with open(filename, "w") as f:
        f.write(f"{m}, {n}, {k}, {j}, {s}: {results}\n")
    return filename  # 返回生成的文件名以便于其他地方使用


class AlgorithmWorker(QThread):
    finished = pyqtSignal(list, list, str)

    def __init__(self, m, n, k, j, s):
        super().__init__()
        self.m = m
        self.n = n
        self.k = k
        self.j = j
        self.s = s
    def run(self):
        n_numbers = random.sample(range(1, self.m + 1), self.n)
        cover_graph = subset_cover_graph(n_numbers, self.k, self.j, self.s)
        result = simulated_annealing(cover_graph, n_numbers)
        params = f"{self.m}-{self.n}-{self.k}-{self.j}-{self.s}"
        self.finished.emit(n_numbers, result, params)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Set Cover Solver')
        self.setGeometry(400, 100, 800, 600)
        self.stacked_widget = QStackedWidget()
        self.init_solver_ui()
        self.init_database_ui()
        self.init_optimal_selection_ui()
        self.init_more3_ui()  # 确保这一行在这里，以初始化 more3 界面
        self.setCentralWidget(self.stacked_widget)
        self.all_data = []  # 初始化数据存储列表
    def init_solver_ui(self):
        self.solver_widget = QWidget()
        solver_layout = QVBoxLayout(self.solver_widget)

        params_layout = QHBoxLayout()
        self.m_input = self.create_combobox(range(45, 101))
        self.n_input = self.create_combobox(range(7, 26))
        self.k_input = self.create_combobox(range(4, 11))
        self.j_input = self.create_combobox([])
        self.s_input = self.create_combobox([])

        self.k_input.currentIndexChanged.connect(self.update_j_options)
        self.j_input.currentIndexChanged.connect(self.update_s_options)

        params_layout.addWidget(QLabel("m:"))
        params_layout.addWidget(self.m_input)
        params_layout.addWidget(QLabel("n:"))
        params_layout.addWidget(self.n_input)
        params_layout.addWidget(QLabel("k:"))
        params_layout.addWidget(self.k_input)
        params_layout.addWidget(QLabel("j:"))
        params_layout.addWidget(self.j_input)
        params_layout.addWidget(QLabel("s:"))
        params_layout.addWidget(self.s_input)
        solver_layout.addLayout(params_layout)

        self.result_text_edit = QTextEdit()
        self.result_text_edit.setReadOnly(True)
        solver_layout.addWidget(self.result_text_edit)

        start_btn = QPushButton('Start', self)
        start_btn.clicked.connect(self.start_thread)
        database_btn = QPushButton('Database', self)
        database_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(start_btn)
        btn_layout.addWidget(database_btn)

        way_btn = QPushButton('more', self)  # 新按钮
        way_btn.clicked.connect(self.show_optimal_selection_ui)  # 连接到新界面显示的方法
        btn_layout.addWidget(way_btn)

        solver_layout.addLayout(btn_layout)
        self.stacked_widget.addWidget(self.solver_widget)


    def init_optimal_selection_ui(self):
        self.optimal_selection_widget = QWidget()
        optimal_selection_layout = QVBoxLayout(self.optimal_selection_widget)

        # 添加标题
        title_label = QLabel('Custom filtering')
        title_label.setAlignment(Qt.AlignCenter)
        optimal_selection_layout.addWidget(title_label)

        # 第一行选择框
        selection_layout = QHBoxLayout()
        self.position_combobox1 = self.create_number_combobox(1, 6)
        self.position_combobox2 = self.create_number_combobox(1, 6)
        self.position_combobox3 = self.create_number_combobox(1, 6)
        selection_layout.addWidget(self.position_combobox1)
        selection_layout.addWidget(self.position_combobox2)
        selection_layout.addWidget(self.position_combobox3)
        optimal_selection_layout.addLayout(selection_layout)

        # 第二行输入框
        input_layout = QHBoxLayout()
        self.input_number1 = QLineEdit()
        self.input_number2 = QLineEdit()
        self.input_number3 = QLineEdit()
        input_layout.addWidget(self.input_number1)
        input_layout.addWidget(self.input_number2)
        input_layout.addWidget(self.input_number3)
        optimal_selection_layout.addLayout(input_layout)

        # 添加文件夹选择按钮和输入框
        folder_selection_layout = QHBoxLayout()
        self.folder_input = QLineEdit()
        select_folder_btn = QPushButton("Select Folder")
        select_folder_btn.clicked.connect(self.select_folder)
        folder_selection_layout.addWidget(self.folder_input)
        folder_selection_layout.addWidget(select_folder_btn)

        optimal_selection_layout.addLayout(folder_selection_layout)

        # 筛选按钮
        filter_button = QPushButton('Filter Results')
        filter_button.clicked.connect(self.filter_results)
        optimal_selection_layout.addWidget(filter_button)

        # 原始数列展示区域
        self.original_display = QTextBrowser()
        self.original_display.setPlainText("Original Data Loading...")
        optimal_selection_layout.addWidget(self.original_display)

        # 筛选结果展示区域
        self.results_display = QTextBrowser()
        self.results_display.setPlainText("Filtered Results Appear Here...")
        optimal_selection_layout.addWidget(self.results_display)

        self.stacked_widget.addWidget(self.optimal_selection_widget)

        self.load_original_data()
        back_btn = QPushButton('Back', self)  # 按钮文本设置为 "Back"
        back_btn.clicked.connect(self.show_solver_ui)  # 点击后调用 show_solver_ui 函数
        optimal_selection_layout.addWidget(back_btn)

        self.stacked_widget.addWidget(self.optimal_selection_widget)

        button_layout = QHBoxLayout()
        optimal_selection_layout.addLayout(button_layout)

        # 添加新的 "more2" 按钮
        more2_btn = QPushButton('more2', self)
        more2_btn.clicked.connect(self.show_more2_ui)
        button_layout.addWidget(more2_btn)

        self.stacked_widget.addWidget(self.optimal_selection_widget)

    def init_more2_ui(self):
        self.more2_widget = QWidget()
        more2_layout = QVBoxLayout(self.more2_widget)

        title_label = QLabel('Range filtering', self.more2_widget)
        title_label.setAlignment(Qt.AlignCenter)
        more2_layout.addWidget(title_label)

        # 第一行选择框
        selection_layout = QHBoxLayout()
        self.position_combobox1_more2 = self.create_number_combobox(1, 6)
        self.position_combobox2_more2 = self.create_number_combobox(1, 6)
        self.position_combobox3_more2 = self.create_number_combobox(1, 6)
        selection_layout.addWidget(self.position_combobox1_more2)
        selection_layout.addWidget(self.position_combobox2_more2)
        selection_layout.addWidget(self.position_combobox3_more2)
        more2_layout.addLayout(selection_layout)

        # 第二行输入范围，使用两个 QLineEdit
        input_layout = QHBoxLayout()
        self.input_range1_min = QLineEdit()
        self.input_range1_max = QLineEdit()
        self.input_range2_min = QLineEdit()
        self.input_range2_max = QLineEdit()
        self.input_range3_min = QLineEdit()
        self.input_range3_max = QLineEdit()

        input_layout.addWidget(self.input_range1_min)
        input_layout.addWidget(QLabel("to"))
        input_layout.addWidget(self.input_range1_max)
        input_layout.addWidget(self.input_range2_min)
        input_layout.addWidget(QLabel("to"))
        input_layout.addWidget(self.input_range2_max)
        input_layout.addWidget(self.input_range3_min)
        input_layout.addWidget(QLabel("to"))
        input_layout.addWidget(self.input_range3_max)
        more2_layout.addLayout(input_layout)

        # 文件选择按钮和输入框
        folder_selection_layout = QHBoxLayout()
        self.folder_input_more2 = QLineEdit()
        select_folder_btn_more2 = QPushButton("Select File")
        select_folder_btn_more2.clicked.connect(self.select_folder_more2)
        folder_selection_layout.addWidget(self.folder_input_more2)
        folder_selection_layout.addWidget(select_folder_btn_more2)
        more2_layout.addLayout(folder_selection_layout)

        # 筛选按钮
        filter_button = QPushButton('Filter Results')
        filter_button.clicked.connect(self.filter_results_range_more2)
        more2_layout.addWidget(filter_button)

        # 原始数据展示区域
        self.original_display_more2 = QTextBrowser()
        self.original_display_more2.setPlainText("Original Data Loading...")
        more2_layout.addWidget(self.original_display_more2)

        # 筛选结果展示区域
        self.results_display_more2 = QTextBrowser()
        self.results_display_more2.setPlainText("Filtered Results Appear Here...")
        more2_layout.addWidget(self.results_display_more2)

        button_layout = QHBoxLayout()
        back_btn_more2 = QPushButton('Back', self.more2_widget)
        back_btn_more2.clicked.connect(self.show_previous_ui)
        button_layout.addWidget(back_btn_more2)

        # 添加新的 "more3" 按钮
        more3_btn = QPushButton('more3', self.more2_widget)
        more3_btn.clicked.connect(self.show_more3_ui)
        button_layout.addWidget(more3_btn)

        more2_layout.addLayout(button_layout)

        self.stacked_widget.addWidget(self.more2_widget)

    def show_previous_ui(self):
        # 如果你已知返回的具体界面，可以直接设置索引
        self.stacked_widget.setCurrentIndex(0)  # 假设主界面的索引是 0

    def select_folder_more2(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Select a text file", "",
                                                   "Text Files (*.txt)", options=options)
        if file_name:
            self.folder_input_more2.setText(file_name)
            self.load_file_data_more2(file_name)

    def filter_results_range_more2(self):
        try:
            positions = [
                int(self.position_combobox1_more2.currentText()) - 1,
                int(self.position_combobox2_more2.currentText()) - 1,
                int(self.position_combobox3_more2.currentText()) - 1
            ]
            ranges = [
                (int(self.input_range1_min.text()), int(self.input_range1_max.text())),
                (int(self.input_range2_min.text()), int(self.input_range2_max.text())),
                (int(self.input_range3_min.text()), int(self.input_range3_max.text()))
            ]
        except ValueError as e:
            self.results_display_more2.setPlainText(f"Input error: {e}")
            return

        filtered_results = []
        for line in self.all_data_more2:
            try:
                _, data_str = line.split(':')
                data = eval(data_str.strip())
                for tuple in data:
                    if all(r[0] <= tuple[pos] <= r[1] for pos, r in zip(positions, ranges)):
                        filtered_results.append(tuple)
            except Exception as e:
                self.results_display_more2.setPlainText(f"Error parsing data: {str(e)}")
                return

        if filtered_results:
            result_display = "\n".join(', '.join(map(str, res)) for res in filtered_results)
            self.results_display_more2.setPlainText(result_display)
        else:
            self.results_display_more2.setPlainText("No results found within specified ranges.")

    def show_more2_ui(self):
        # 确保已经初始化了 more2 界面
        if not hasattr(self, 'more2_widget'):
            self.init_more2_ui()

        # 设置堆栈窗口小部件以显示新界面
        self.stacked_widget.setCurrentWidget(self.more2_widget)

    def init_more3_ui(self):
            self.more3_widget = QWidget()
            more3_layout = QVBoxLayout(self.more3_widget)

            title_label = QLabel('Expectation filtering', self.more3_widget)
            title_label.setAlignment(Qt.AlignCenter)
            more3_layout.addWidget(title_label)

            # 第一行选择框
            selection_layout = QHBoxLayout()
            self.position_combobox1_more3 = self.create_number_combobox(1, 6)
            self.position_combobox2_more3 = self.create_number_combobox(1, 6)
            self.position_combobox3_more3 = self.create_number_combobox(1, 6)
            selection_layout.addWidget(self.position_combobox1_more3)
            selection_layout.addWidget(self.position_combobox2_more3)
            selection_layout.addWidget(self.position_combobox3_more3)
            more3_layout.addLayout(selection_layout)

            # 第二行输入期望值
            input_layout = QHBoxLayout()
            self.target_sum_input = QLineEdit()
            input_layout.addWidget(QLabel("Target Sum:"))
            input_layout.addWidget(self.target_sum_input)
            more3_layout.addLayout(input_layout)

            folder_selection_layout = QHBoxLayout()
            self.folder_input_more3 = QLineEdit()
            select_folder_btn_more3 = QPushButton("Select File")
            select_folder_btn_more3.clicked.connect(self.select_folder_more3)
            folder_selection_layout.addWidget(self.folder_input_more3)
            folder_selection_layout.addWidget(select_folder_btn_more3)
            more3_layout.addLayout(folder_selection_layout)

            # 筛选按钮
            filter_button = QPushButton('Filter Results')
            filter_button.clicked.connect(self.filter_results_more3)
            more3_layout.addWidget(filter_button)

            self.original_display_more3 = QTextBrowser()
            self.original_display_more3.setPlainText("Original Data Loading...")
            more3_layout.addWidget(self.original_display_more3)

            # 结果展示区域
            self.results_display_more3 = QTextBrowser()
            self.results_display_more3.setPlainText("Results will appear here...")
            more3_layout.addWidget(self.results_display_more3)

            # 返回按钮
            back_btn_more3 = QPushButton('Back', self.more3_widget)
            back_btn_more3.clicked.connect(self.show_previous_ui)
            more3_layout.addWidget(back_btn_more3)

            self.stacked_widget.addWidget(self.more3_widget)

    def select_folder_more3(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Select a text file", "",
                                                   "Text Files (*.txt)", options=options)
        if file_name:
            self.folder_input_more3.setText(file_name)
            self.load_file_data_more3(file_name)

    def load_file_data_more3(self, file_path):
        try:
            with open(file_path, 'r') as file:
                self.all_data = file.readlines()
            self.original_display_more3.setPlainText(''.join(self.all_data))
        except FileNotFoundError:
            self.original_display_more3.setPlainText("Selected file not found.")

    def filter_results_more3(self):
        try:
            positions = [
                int(self.position_combobox1_more3.currentText()) - 1,
                int(self.position_combobox2_more3.currentText()) - 1,
                int(self.position_combobox3_more3.currentText()) - 1
            ]
            target_sum = int(self.target_sum_input.text())
        except ValueError:
            self.results_display_more3.setPlainText("Please ensure all inputs are valid integers.")
            return

        filtered_results = []
        for line in self.all_data:
            try:
                _, data_str = line.split(':')
                data = eval(data_str.strip())
                for tuple in data:
                    if sum(tuple[pos] for pos in positions) == target_sum:
                        filtered_results.append(tuple)
            except Exception as e:
                self.results_display_more3.setPlainText(f"Error parsing data: {str(e)}")
                return

        if filtered_results:
            result_display = "\n".join(str(res) for res in filtered_results)
            self.results_display_more3.setPlainText(result_display)
        else:
            self.results_display_more3.setPlainText("No results found matching the target sum.")

    def show_more3_ui(self):
        if not hasattr(self, 'more3_widget'):
            self.init_more3_ui()  # 如果 more3_widget 未初始化，则先初始化
        self.stacked_widget.setCurrentWidget(self.more3_widget)

    def show_solver_ui(self):
        # 切换到求解界面
        self.stacked_widget.setCurrentIndex(0)

    def select_folder(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Select a text file", "",
                                                   "Text Files (*.txt)", options=options)
        if file_name:
            self.folder_input.setText(file_name)
            self.load_file_data(file_name)

    def load_file_data(self, file_path):
        try:
            with open(file_path, 'r') as file:
                self.all_data = file.readlines()
            self.original_display.setPlainText(''.join(self.all_data))
        except FileNotFoundError:
            self.original_display.setPlainText("Selected file not found.")

    def load_file_data_more2(self, file_path):
        try:
            with open(file_path, 'r') as file:
                self.all_data_more2 = file.readlines()
            self.original_display_more2.setPlainText(''.join(self.all_data_more2))
        except FileNotFoundError:
            self.original_display_more2.setPlainText("Selected file not found.")


    def show_optimal_selection_ui(self):
        # 显示新界面
        self.stacked_widget.setCurrentWidget(self.optimal_selection_widget)

    def create_number_combobox(self, min_value, max_value):
        combobox = QComboBox()
        for value in range(min_value, max_value + 1):
            combobox.addItem(str(value))
        return combobox

    def load_original_data(self):
        result_files = glob.glob('*_*_*_*_*.txt')
        self.all_data = []  # 清除现有数据
        for filename in result_files:
            try:
                with open(filename, 'r') as file:
                    file_data = file.read().strip().split('\n')
                    for line in file_data:
                        if line:
                            self.all_data.append(line)  # 确保添加的是字符串
            except FileNotFoundError:
                continue  # 如果文件未找到，跳过

        if self.all_data:
            self.original_display.setPlainText("\n\n".join(self.all_data))
        else:
            self.original_display.setPlainText("未找到结果文件。")

    def filter_results(self):
        try:
            # 从用户输入中提取位置和目标数字
            positions = [
                int(self.position_combobox1.currentText()) - 1,
                int(self.position_combobox2.currentText()) - 1,
                int(self.position_combobox3.currentText()) - 1
            ]
            target_numbers = [
                int(self.input_number1.text()),
                int(self.input_number2.text()),
                int(self.input_number3.text())
            ]
        except ValueError:
            self.results_display.setPlainText("请在所有字段中输入有效数字。")
            return


        filtered_results = []
        for line in self.all_data:
            try:
                # 从每一行中解析出数据部分
                _, data_str = line.split(':')
                data = eval(data_str.strip())  # 安全风险，仅在可信数据上使用
                # 筛选匹配的序列
                for tuple in data:
                    if all(tuple[pos] == num for pos, num in zip(positions, target_numbers)):
                        filtered_results.append(tuple)
            except Exception as e:
                self.results_display.setPlainText(f"解析错误: {str(e)}")
                return

        if filtered_results:
            result_display = "\n".join(', '.join(map(str, res)) for res in filtered_results)
            self.results_display.setPlainText(result_display)
        else:
            self.results_display.setPlainText("未找到符合条件的结果。")

    def init_database_ui(self):
        self.database_widget = QWidget()
        database_layout = QVBoxLayout(self.database_widget)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.button_widget = QWidget()
        self.button_layout = QVBoxLayout()
        self.button_widget.setLayout(self.button_layout)
        self.scroll_area.setWidget(self.button_widget)
        database_layout.addWidget(self.scroll_area)

        self.display_text = QTextBrowser()
        database_layout.addWidget(self.display_text)

        back_btn = QPushButton('Back to Solver', self)
        back_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        database_layout.addWidget(back_btn)

        self.stacked_widget.addWidget(self.database_widget)
        self.load_data()

    def load_data(self):
        clear_layout(self.button_layout)
        try:
            with open("Database.txt", "r") as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    params, result = line.split(': ', 1)
                    btn = QPushButton(params, self)
                    btn.clicked.connect(lambda checked, text=result: self.display_text.setText(text))
                    self.button_layout.addWidget(btn)
        except FileNotFoundError:
            self.display_text.setText("Database.txt not found.")



    def create_combobox(self, values):
        combobox = QComboBox()
        combobox.addItem("")  # Empty default option
        for value in values:
            combobox.addItem(str(value))
        combobox.setFixedWidth(150)
        return combobox

    def update_result(self, n_numbers, result, params):
        self.all_data.append(result)  # 添加新结果到列表
        result_display = "\n".join(f"Combination {idx + 1}: {comb}" for idx, comb in enumerate(result))

        # 解析参数
        m, n, k, j, s = params.split('-')
        # 保存结果并获取实际保存的文件名
        actual_filename = save_to_database(int(m), int(n), int(k), int(j), int(s), result)

        # 在文本框中显示结果和文件名
        self.result_text_edit.setText(
            f"Randomly selected n={len(n_numbers)} numbers: {n_numbers}\n\nThe approximate minimal set cover of k samples combinations found:\n{result_display}\n\nFile saved as: {actual_filename}")
    def start_thread(self):
        m = int(self.m_input.currentText()) if self.m_input.currentText() else 0
        n = int(self.n_input.currentText()) if self.n_input.currentText() else 0
        k = int(self.k_input.currentText()) if self.k_input.currentText() else 0
        j = int(self.j_input.currentText()) if self.j_input.currentText() else 0
        s = int(self.s_input.currentText()) if self.s_input.currentText() else 0

        if m > 0 and n > 0 and k > 0 and j > 0 and s > 0:
            self.worker = AlgorithmWorker(m, n, k, j, s)
            self.worker.finished.connect(self.update_result)
            self.worker.start()

    def update_j_options(self):
        self.j_input.clear()
        k = int(self.k_input.currentText()) if self.k_input.currentText() else 0
        self.j_input.addItems([str(i) for i in range(1, k+1)])
        self.update_s_options()

    def update_s_options(self):
        self.s_input.clear()
        j = int(self.j_input.currentText()) if self.j_input.currentText() else 0
        self.s_input.addItems([str(i) for i in range(1, j+1)])

def clear_layout(layout):
    while layout.count():
        child = layout.takeAt(0)
        if child.widget():
            child.widget().deleteLater()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())