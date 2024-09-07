import pandas as pd
import matplotlib.pyplot as plt

# 定义Excel文件的路径

file_paths = [
    'C:/Users/Zhang Xi/Desktop/Model1/loss.xlsx',
    'C:/Users/Zhang Xi/Desktop/Model2/loss.xlsx',

    'C:/Users/Zhang Xi/Desktop/Model4/loss.xlsx',
    'C:/Users/Zhang Xi/Desktop/Model5/loss.xlsx'
]
#'C:/Users/Zhang Xi/Desktop/Model3/loss.xlsx',
# 定义存储数据的字典
train_loss_data = {}
test_loss_data = {}

# 给每个文件一个标签
labels = ['GRU', 'GRU&CNN', 'Transformer','Transformer&CNN']

# 遍历每个文件，加载数据
for i, file_path in enumerate(file_paths):
    df = pd.read_excel(file_path)
    train_loss_data[labels[i]] = df['Train Loss']
    test_loss_data[labels[i]] = df['Test Loss']

# 绘制Train Loss对比图
plt.figure(figsize=(10, 6))
for name, loss in train_loss_data.items():
    plt.plot(loss, label=f'{name}')
plt.title('Train Loss Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('train_loss_comparison.png')


# 绘制Test Loss对比图
plt.figure(figsize=(10, 6))
for name, loss in test_loss_data.items():
    plt.plot(loss, label=f'{name}')
plt.title('Test Loss Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('test_loss_comparison.png')




