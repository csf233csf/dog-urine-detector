import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from visualdl import LogWriter
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 搭建一个CNN + FC + LSTM模型，使用传感器数据训练+预测狗是否在排尿
# Construct a CNN + FC + LSTM model for training and predicting whether a dog is urinating based on sensor data
class CNNSensorTaggingModel(nn.Layer):
    def __init__(self):
        super(CNNSensorTaggingModel, self).__init__()
        self.conv1 = nn.Conv1D(in_channels=1, out_channels=16, kernel_size=3, padding='same')
        self.conv2 = nn.Conv1D(in_channels=16, out_channels=32, kernel_size=3, padding='same')
        self.pool = nn.MaxPool1D(kernel_size=2)
        self.fc1 = nn.Linear(32 * 21, 128)  # 全连接层连接CNN输出和LSTM输入
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2, dropout=0.5)
        self.fc2 = nn.Linear(64, 2)  # 输出层，2个类别：排尿或不排尿
        self.init_weights()

    # 初始化模型参数
    # Initialize model parameters
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.initializer.XavierUniform()(param)
            elif 'bias' in name:
                nn.initializer.Constant(0.0)(param)

    # 定义前向传播过程
    # Define the forward pass
    def forward(self, x):
        x = F.relu(self.conv1(x))  # 卷积和ReLU激活
        x = self.pool(F.relu(self.conv2(x)))  # 卷积、ReLU激活和池化
        x = x.reshape([x.shape[0], -1])  # 展平数据
        x = F.relu(self.fc1(x))  # 全连接层和ReLU激活
        x = x.unsqueeze(1)  # 为LSTM添加序列维度
        x, _ = self.lstm(x)  # LSTM层
        x = self.fc2(x[:, -1, :])  # 输出最后的分类结果
        return x

# 加载和预处理数据
# Load and preprocess data
train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data.csv')

X_train = train_df.drop(columns=['label']).values
y_train = train_df['label'].values
X_test = test_df.drop(columns=['label']).values
y_test = test_df['label'].values

# 检查数据是否有缺失值，并且转换数据的形状，保证input长度一致
# Check for missing values and standardize the data
assert not np.any(np.isnan(X_train)), "Training data contains NaN values"
assert not np.any(np.isnan(X_test)), "Testing data contains NaN values"

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = X_train.reshape(-1, 1, 42)
X_test = X_test.reshape(-1, 1, 42)

# 创建customized loader，加载数据
# Create a dataset class and use it to create data loaders
class CustomDataset(paddle.io.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label

batch_size = 32
train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)
train_loader = paddle.io.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = paddle.io.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型、优化器和损失函数
# Initialize the model, optimizer, and loss function
model = CNNSensorTaggingModel()
criterion = nn.CrossEntropyLoss()
optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())

# 创建日志记录器，用于记录训练过程中的数据
# Create a log writer to record data during training
log_writer = LogWriter(logdir="./log")

# 训练循环，包含30个周期
# Training loop for 30 epochs
num_epochs = 30

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

    for batch_x, batch_y in progress_bar:
        batch_x = paddle.to_tensor(batch_x, dtype='float32')
        batch_y = paddle.to_tensor(batch_y, dtype='int64')

        outputs = model(batch_x)  # 前向传播 forwarding
        loss = criterion(outputs, batch_y)  # 计算损失
        
        loss.backward()  # 反向传播
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度剪裁，尽量不overfit
        optimizer.step()  # 更新参数
        optimizer.clear_grad()  # 清除梯度

        epoch_loss += loss.numpy().item()  # 累加损失
        predictions = outputs.argmax(axis=-1)  # 获取预测值
        correct_predictions += (predictions == batch_y).sum().item()  # 计算正确预测的数量
        total_predictions += batch_y.shape[0]  # 计算总预测数量
        progress_bar.set_postfix({"loss": loss.numpy().item(), "accuracy": correct_predictions / total_predictions})

    # 记录训练损失和准确率到日志
    # Record training loss and accuracy to the log
    log_writer.add_scalar(tag="train/loss", step=epoch, value=epoch_loss / len(train_loader))
    log_writer.add_scalar(tag="train/accuracy", step=epoch, value=correct_predictions / total_predictions)

    # 评估模型在测试数据集上的表现
    # Evaluate the model on the test dataset
    model.eval()
    with paddle.no_grad():
        test_loss = 0.0
        test_correct_predictions = 0
        test_total_predictions = 0
        all_predictions = []
        all_labels = []
        
        for batch_x, batch_y in test_loader:
            batch_x = paddle.to_tensor(batch_x, dtype='float32')
            batch_y = paddle.to_tensor(batch_y, dtype='int64')

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            test_loss += loss.numpy().item()

            test_predictions = outputs.argmax(axis=-1)
            test_correct_predictions += (test_predictions == batch_y).sum().item()
            test_total_predictions += batch_y.shape[0]

            all_predictions.extend(test_predictions.numpy())
            all_labels.extend(batch_y.numpy())

        # 计算和打印测试准确率
        # Calculate and print test accuracy
        test_accuracy = accuracy_score(all_labels, all_predictions)
        print(f"Test Accuracy: {test_accuracy:.4f}")

        # 记录测试损失和准确率到日志
        # Record test loss and accuracy to the log
        log_writer.add_scalar(tag="test/loss", step=epoch, value=test_loss / len(test_loader))
        log_writer.add_scalar(tag="test/accuracy", step=epoch, value=test_accuracy)

    # 打印每个epoch的训练和测试结果
    # Print training and test results for each epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss / len(train_loader):.4f}, "
          f"Train Accuracy: {correct_predictions / total_predictions:.4f}, "
          f"Test Loss: {test_loss / len(test_loader):.4f}, "
          f"Test Accuracy: {test_accuracy:.4f}")

# 保存训练好的模型参数
# Save the trained model
paddle.jit.save(model, 'saved_inference_model/model', input_spec=[paddle.static.InputSpec(shape=[None, 1, 42], dtype='float32')])

# 可视化训练数据
log_writer.add_histogram(tag='train_data', values=X_train.flatten(), step=0, buckets=100)
log_writer.add_histogram(tag='test_data', values=X_test.flatten(), step=0, buckets=100)
# 高维数据
y_train_str = [str(label) for label in y_train]
y_test_str = [str(label) for label in y_test]
log_writer.add_embeddings(tag='train_embeddings', mat=X_train.reshape(X_train.shape[0], -1), metadata=y_train_str, labels=y_train)
log_writer.add_embeddings(tag='test_embeddings', mat=X_test.reshape(X_test.shape[0], -1), metadata=y_test_str, labels=y_test)

log_writer.close()
