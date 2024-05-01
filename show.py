import random, glob, datetime, sys, itertools, math, time
from collections import defaultdict
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QTextEdit, QLineEdit, QComboBox, QHBoxLayout, QScrollArea, QTextBrowser,  QStackedWidget, QLabel
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

def save_to_database(params, results):
    filename = f"Result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(filename, "w") as f:
        f.write(f"{params}: {results}\n")

class AlgorithmWorker(QThread):
    finished = pyqtSignal(list, list, str)

    def __init__(self, n_numbers, k, j, s):
        super().__init__()
        self.n_numbers = n_numbers
        self.k = k
        self.j = j
        self.s = s

    def run(self):
        cover_graph = subset_cover_graph(self.n_numbers, self.k, self.j, self.s)
        result = simulated_annealing(cover_graph, self.n_numbers)
        params = f"{len(self.n_numbers)}-{self.k}-{self.j}-{self.s}"
        self.finished.emit(self.n_numbers, result, params)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Set Cover Solver')
        self.setGeometry(400, 100, 800, 600)
        self.stacked_widget = QStackedWidget()
        self.init_solver_ui()
        self.init_database_ui()
        self.init_optimal_selection_ui()
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

        way_btn = QPushButton('way', self)  # 新按钮
        way_btn.clicked.connect(self.show_optimal_selection_ui)  # 连接到新界面显示的方法
        btn_layout.addWidget(way_btn)

        solver_layout.addLayout(btn_layout)
        self.stacked_widget.addWidget(self.solver_widget)

    def init_optimal_selection_ui(self):
        self.optimal_selection_widget = QWidget()
        optimal_selection_layout = QVBoxLayout(self.optimal_selection_widget)

        # 添加标题
        title_label = QLabel('AN OPTIMAL SAMPLES SELECTION SYSTEM')
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

    def show_optimal_selection_ui(self):
        # 显示新界面
        self.stacked_widget.setCurrentWidget(self.optimal_selection_widget)

    def create_number_combobox(self, min_value, max_value):
        combobox = QComboBox()
        for value in range(min_value, max_value + 1):
            combobox.addItem(str(value))
        return combobox

    def load_original_data(self):
        # 匹配当前目录下所有以'Result_'开头的txt文件
        result_files = glob.glob('Result_*.txt')
        print("找到的文件：", result_files)  # 调试行，检查找到了哪些文件
        all_data = []
        for filename in result_files:
            try:
                with open(filename, 'r') as file:
                    all_data.append(file.read())
            except FileNotFoundError:
                print("未找到文件：", filename)  # 调试行，如果文件未找到
                continue

        if all_data:
            self.original_display.setPlainText("\n\n".join(all_data))
        else:
            self.original_display.setPlainText("未找到结果文件。")

    def filter_results(self):
        try:
            # 从界面获取位置和数字
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
            # 如果输入不是有效的数字，显示错误消息
            self.results_display.setPlainText("Please enter valid numbers in all fields.")
            return

        filtered_results = []
        # 遍历所有数据集，这假设`all_data`是正确加载的数据列表
        for data_string in self.all_data:
            try:
                # 将数据字符串转换为整数列表
                numbers = list(map(int, data_string.split(', ')))
                # 检查每个指定位置的数字是否匹配用户输入
                if all(numbers[pos] == target for pos, target in zip(positions, target_numbers)):
                    filtered_results.append(numbers)
            except ValueError:
                continue  # 如果转换失败，跳过这个数据集

        if filtered_results:
            # 如果找到匹配的数据集，格式化并显示
            result_display = "\n".join(', '.join(map(str, res)) for res in filtered_results)
            self.results_display.setPlainText(result_display)
        else:
            self.results_display.setPlainText("No results found matching the criteria.")

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

    def start_thread(self):
        m = int(self.m_input.currentText()) if self.m_input.currentText() else 0
        n = int(self.n_input.currentText()) if self.n_input.currentText() else 0
        k = int(self.k_input.currentText()) if self.k_input.currentText() else 0
        j = int(self.j_input.currentText()) if self.j_input.currentText() else 0
        s = int(self.s_input.currentText()) if self.s_input.currentText() else 0

        if m > 0 and n > 0 and k > 0 and j > 0 and s > 0:
            n_numbers = random.sample(range(1, m + 1), n)
            self.worker = AlgorithmWorker(n_numbers, k, j, s)
            self.worker.finished.connect(self.update_result)
            self.worker.start()

    def update_result(self, n_numbers, result, params):
        self.all_data.append(result)  # 添加新结果到列表
        result_display = "\n".join(f"Combination {idx + 1}: {comb}" for idx, comb in enumerate(result))
        self.result_text_edit.setText(
            f"Randomly selected n={len(n_numbers)} numbers: {n_numbers}\n\nThe approximate minimal set cover of k samples combinations found:\n{result_display}")
        save_to_database(params, result)

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