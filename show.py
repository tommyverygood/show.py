import random, glob, datetime, sys, itertools, math, time
from collections import defaultdict, deque
from functools import lru_cache
import concurrent.futures
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout,
                             QWidget, QTextEdit, QLineEdit, QComboBox, QHBoxLayout,
                             QScrollArea, QTextBrowser, QStackedWidget, QLabel, QFileDialog, QCheckBox, QMessageBox)

from PyQt5.QtCore import pyqtSignal, QThread, Qt


@lru_cache(None)
def generate_combinations(elements, k):
    return list(itertools.combinations(elements, k))


def preprocess_subsets(j_subsets, k_combinations, s):
    subset_dict = defaultdict(set)
    for k_comb in k_combinations:
        k_set = set(k_comb)
        # 只考虑有高重叠或相关性的子集
        relevant_j_subsets = filter(lambda j_sub: len(set(j_sub) & k_set) >= s - 1, j_subsets)
        for j_sub in relevant_j_subsets:
            s_combinations = itertools.combinations(j_sub, s)
            if any(set(subset).issubset(k_set) for subset in s_combinations):
                subset_dict[tuple(k_comb)].add(tuple(j_sub))
    return subset_dict


def subset_cover_graph(n_numbers, k, j, s):
    j_subsets = generate_combinations(tuple(n_numbers), j)
    k_combinations = generate_combinations(tuple(n_numbers), k)
    return preprocess_subsets(j_subsets, k_combinations, s)


def evaluate_solution(solution, cover_graph):
    covered_j_subsets = set(itertools.chain(*(cover_graph[k] for k in solution)))
    return len(covered_j_subsets)


def modify_solution(current_solution, cover_graph):
    new_solution = current_solution[:]
    if random.random() > 0.5 and len(new_solution) > 1:
        new_solution.remove(random.choice(new_solution))
    else:
        k_combinations = list(cover_graph.keys())
        possible_additions = [k for k in k_combinations if k not in new_solution]
        if possible_additions:
            new_solution.append(random.choice(possible_additions))
    return new_solution


def generate_good_neighbors(current_solution, cover_graph, current_score):
    neighbors = []
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
    futures = [executor.submit(modify_solution, current_solution, cover_graph) for _ in range(10)]
    for future in concurrent.futures.as_completed(futures):
        new_solution = future.result()
        neighbor_score = evaluate_solution(new_solution, cover_graph)
        if new_solution not in neighbors and new_solution != current_solution and neighbor_score > current_score:
            neighbors.append(new_solution)
    return neighbors


def tabu_search(cover_graph, max_iterations=1000, tabu_size=10, max_tabu_tenure=5):
    current_solution = initial_heuristic_solution(cover_graph)
    best_solution = current_solution
    tabu_list = deque(maxlen=tabu_size)
    best_score = evaluate_solution(best_solution, cover_graph)
    current_score = best_score
    iteration = 0
    scores = {tuple(current_solution): best_score}

    while iteration < max_iterations:
        neighbors = generate_good_neighbors(current_solution, cover_graph, current_score)
        neighbors = [neighbor for neighbor in neighbors if tuple(neighbor) not in tabu_list]
        if not neighbors:
            break
        best_neighbor = max(neighbors, key=lambda neighbor: scores.setdefault(tuple(neighbor),
                                                                              evaluate_solution(neighbor, cover_graph)))
        neighbor_score = scores[tuple(best_neighbor)]

        if neighbor_score > current_score or tuple(best_neighbor) not in tabu_list:
            current_solution = best_neighbor
            current_score = neighbor_score
            tabu_list.append(tuple(best_neighbor))

        if neighbor_score > best_score:
            best_solution = best_neighbor
            best_score = neighbor_score

        tabu_list = deque((tabu_item, tenure - 1) for tabu_item, tenure in tabu_list if tenure - 1 > 0)
        iteration += 1

    return best_solution


def initial_heuristic_solution(cover_graph):
    solution = []
    uncovered_j_subsets = set(itertools.chain(*cover_graph.values()))
    while uncovered_j_subsets:
        best_k_comb = max(cover_graph, key=lambda k: len(set(cover_graph[k]) & uncovered_j_subsets))
        solution.append(best_k_comb)
        uncovered_j_subsets -= set(cover_graph[best_k_comb])
    return solution


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
        start_time = time.time()
        cover_graph = subset_cover_graph(self.n, self.k, self.j, self.s)
        solution = tabu_search(cover_graph)
        end_time = time.time()  # 算法结束后记录时间
        elapsed_time = end_time - start_time  # 计算运行时间
        params = f"{self.m}-{len(self.n)}-{self.k}-{self.j}-{self.s}-{elapsed_time}"
        self.finished.emit(self.n, solution, params)


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
        self.random_generation_needed = False
        self.random_generated = False

    def init_solver_ui(self):
        self.solver_widget = QWidget()
        solver_layout = QVBoxLayout(self.solver_widget)

        params_layout = QHBoxLayout()
        self.m_input = self.create_combobox(range(45, 101))
        self.n_input = self.create_combobox(range(7, 26))
        self.k_input = self.create_combobox(range(4, 11))
        self.j_input = self.create_combobox([])
        self.s_input = self.create_combobox([])

        # Number input fields
        self.number_inputs = [QLineEdit(self) for _ in range(25)]
        self.random_checkbox = QCheckBox("Randomly Generate", self)
        self.random_checkbox.setChecked(False)
        self.random_checkbox.toggled.connect(self.toggle_number_inputs)

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
        params_layout.addWidget(self.random_checkbox)

        number_inputs_layout = QHBoxLayout()
        for number_input in self.number_inputs:
            number_input.setFixedWidth(40)
            number_input.setEnabled(False)
            number_inputs_layout.addWidget(number_input)

        solver_layout.addLayout(params_layout)
        self.n_input.currentIndexChanged.connect(self.enable_number_inputs_based_on_n)

        solver_layout.addLayout(number_inputs_layout)

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

    def enable_number_inputs_based_on_n(self):
        n_selected = int(self.n_input.currentText())
        m_value = int(self.m_input.currentText()) if self.m_input.currentText() else 0

        for i, number_input in enumerate(self.number_inputs):
            if i < n_selected:
                number_input.setEnabled(True)
                number_input.setStyleSheet("color: black; background-color: white;")
                if self.random_checkbox.isChecked() and not self.random_generated:
                    number_input.setText(str(random.randint(1, m_value)))
            else:
                number_input.setEnabled(False)
                number_input.setStyleSheet("color: white; background-color: black;")
                number_input.clear()

    def toggle_number_inputs(self, checked):
        m_text = self.m_input.currentText()
        n_text = self.n_input.currentText()

        if checked and (not m_text or not n_text or not m_text.isdigit() or not n_text.isdigit()):
            QMessageBox.warning(self, "Parameter Error", "Please select both m and n before randomly generating.",
                                QMessageBox.Ok)
            self.random_checkbox.setChecked(False)
            return

        self.random_generation_needed = checked

        # 只有从关闭到开启时生成新的随机数
        if checked and not self.random_generated:
            self.random_generated = True
            n_selected = int(n_text) if n_text.isdigit() else 0
            m_value = int(m_text) if m_text.isdigit() else 0

            for i, number_input in enumerate(self.number_inputs):
                if i < n_selected:
                    number_input.setText(str(random.randint(1, m_value)))

        elif not checked:
            self.random_generated = False

        n_selected = int(n_text) if n_text.isdigit() else 0
        for i, number_input in enumerate(self.number_inputs):
            number_input.setEnabled(not checked if i < n_selected else False)

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
        combobox.setEditable(False)  # 禁止用户自行输入非数字内容
        combobox.setFixedWidth(200)
        return combobox

    def update_result(self, n_numbers, result, params):
        self.all_data.append(result)  # 添加新结果到列表
        result_display = "\n".join(f"Combination {idx + 1}: {comb}" for idx, comb in enumerate(result))

        # 解析参数
        m, n, k, j, s, elapsed_time = params.split('-')
        elapsed_time_str = f"Running time: {float(elapsed_time):.3f} s"
        # 保存结果并获取实际保存的文件名
        actual_filename = save_to_database(int(m), int(n), int(k), int(j), int(s), result)

        # 在文本框中显示结果和文件名
        self.result_text_edit.setText(
            f"Randomly selected n={len(n_numbers)} numbers: {n_numbers}\n\nThe approximate minimal set cover of k samples combinations found:\n{result_display}\n\n{elapsed_time_str}\n\nFile saved as: {actual_filename}")

    def start_thread(self):
        m_text = self.m_input.currentText()
        n_text = self.n_input.currentText()
        k_text = self.k_input.currentText()
        j_text = self.j_input.currentText()
        s_text = self.s_input.currentText()

        # Check if any of the values are empty or not numbers
        if not (m_text.isdigit() and n_text.isdigit() and k_text.isdigit() and
                j_text.isdigit() and s_text.isdigit()):
            QMessageBox.warning(self, "Parameter Error", "Please select all parameters before starting.",
                                QMessageBox.Ok)
            return

        m = int(m_text)
        n = int(n_text)
        k = int(k_text)
        j = int(j_text)
        s = int(s_text)

        # Collect numbers from input fields
        n_values = []
        for input_field in self.number_inputs[:n]:
            if input_field.text().isdigit():
                n_values.append(int(input_field.text()))
            else:
                QMessageBox.warning(self, "Input Error", f"Please fill in all {n} input fields before starting.",
                                    QMessageBox.Ok)
                return

        self.worker = AlgorithmWorker(m, n_values, k, j, s)
        self.worker.finished.connect(self.update_result)
        self.worker.start()

    def update_j_options(self):
        self.j_input.clear()
        k = int(self.k_input.currentText()) if self.k_input.currentText() else 0
        self.j_input.addItems([str(i) for i in range(1, k + 1)])
        self.update_s_options()

    def update_s_options(self):
        self.s_input.clear()
        j = int(self.j_input.currentText()) if self.j_input.currentText() else 0
        self.s_input.addItems([str(i) for i in range(1, j + 1)])


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
