import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error,
                             mean_squared_error,
                             r2_score,
                             accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             classification_report,
                             confusion_matrix)
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import math

# 设置随机种子，确保可重复性
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 超参数设置
EPOCHS = 80
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
HIDDEN_LAYERS = [128, 64]
WEIGHT_DECAY = 5e-4
DROPOUT_RATE = 0.2
DATA_PATH = 'hospital_training_data.csv'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_P0(lambd, mu, s):
    if s * mu <= lambd:
        return None
    rho = lambd / (s * mu)
    sum_part = 0.0
    for k in range(s):
        sum_part += ((lambd / mu) ** k) / math.factorial(k)
    term = ((lambd / mu) ** s) / (math.factorial(s) * (1 - rho))
    return 1 / (sum_part + term)


def compute_metrics(lambd, mu, s):
    P0 = compute_P0(lambd, mu, s)
    if P0 is None:
        return None, None
    rho = lambd / (s * mu)
    numerator = P0 * ((lambd / mu) ** s) * (mu / (s * mu - lambd))
    denominator = math.factorial(s) * (1 - rho) ** 2
    Wq = numerator / denominator
    W_total = Wq + 1 / mu
    return Wq, W_total


class HospitalDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.nurse_targets = torch.LongTensor(targets[:, 0])
        self.doctor_targets = torch.LongTensor(targets[:, 1])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (self.features[idx], (self.nurse_targets[idx], self.doctor_targets[idx]))


class HospitalStaffingModel(nn.Module):
    def __init__(self, input_dim, hidden_layers, n_nurse_classes, n_doctor_classes):
        super().__init__()
        layers = []
        prev_size = input_dim
        for layer_size in hidden_layers:
            layers += [
                nn.Linear(prev_size, layer_size),
                nn.BatchNorm1d(layer_size),
                nn.ReLU(),
                nn.Dropout(DROPOUT_RATE)
            ]
            self._init_weights(layers[-4])
            self._init_weights(layers[-3])
            prev_size = layer_size

        self.shared_layers = nn.Sequential(*layers)
        self.nurse_head = self._create_head(prev_size, n_nurse_classes)
        self.doctor_head = self._create_head(prev_size, n_doctor_classes)

    def _create_head(self, in_features, out_features):
        head = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, out_features)
        )
        self._init_weights(head[0])
        self._init_weights(head[2])
        return head

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self, x):
        s_nurse_max = x[:, -4].long()
        s_doctor_max = x[:, -3].long()

        shared_out = self.shared_layers(x)
        nurse_logits = self.nurse_head(shared_out)
        doctor_logits = self.doctor_head(shared_out)

        batch_size = x.size(0)
        device = x.device

        nurse_mask = torch.zeros_like(nurse_logits, device=device)
        for i in range(batch_size):
            nurse_mask[i, :s_nurse_max[i] + 1] = 1
        nurse_logits = nurse_logits * nurse_mask + (1 - nurse_mask) * -1e9

        doctor_mask = torch.zeros_like(doctor_logits, device=device)
        for i in range(batch_size):
            doctor_mask[i, :s_doctor_max[i] + 1] = 1
        doctor_logits = doctor_logits * doctor_mask + (1 - doctor_mask) * -1e9

        return nurse_logits, doctor_logits


def evaluate_during_training(model, dataloader):
    model.eval()
    all_true_n = []
    all_pred_n = []
    all_true_d = []
    all_pred_d = []

    with torch.no_grad():
        for features, (nurse_t, doctor_t) in dataloader:
            features = features.to(device)
            nurse_logits, doctor_logits = model(features)
            pred_n = torch.argmax(nurse_logits, dim=1).cpu().numpy()
            pred_d = torch.argmax(doctor_logits, dim=1).cpu().numpy()

            all_true_n.extend(nurse_t.numpy().astype(int))
            all_pred_n.extend(pred_n)
            all_true_d.extend(doctor_t.numpy().astype(int))
            all_pred_d.extend(pred_d)

    def calc_metrics(true, pred):
        return {
            'MAE': mean_absolute_error(true, pred),
            'MSE': mean_squared_error(true, pred),
            'R2': r2_score(true, pred)
        }

    return {
        'Nurse': calc_metrics(all_true_n, all_pred_n),
        'Doctor': calc_metrics(all_true_d, all_pred_d)
    }


def final_evaluation(model, dataloader, X_raw_test):
    model.eval()
    all_true_n = []
    all_pred_n = []
    all_true_d = []
    all_pred_d = []

    with torch.no_grad():
        for features, (nurse_t, doctor_t) in dataloader:
            features = features.to(device)
            nurse_logits, doctor_logits = model(features)
            pred_n = torch.argmax(nurse_logits, dim=1).cpu().numpy()
            pred_d = torch.argmax(doctor_logits, dim=1).cpu().numpy()

            all_true_n.extend(nurse_t.numpy().astype(int))
            all_pred_n.extend(pred_n)
            all_true_d.extend(doctor_t.numpy().astype(int))
            all_pred_d.extend(pred_d)

    # 创建结果数据框
    results_df = X_raw_test.reset_index(drop=True).copy()
    results_df['Predicted_Nurses'] = all_pred_n
    results_df['True_Nurses'] = all_true_n
    results_df['Predicted_Doctors'] = all_pred_d
    results_df['True_Doctors'] = all_true_d
    results_df.to_csv('test_predictions.csv', index=False)
    print("\n预测结果已保存至 test_predictions.csv")

    def generate_report(true, pred, role):
        print(f"\n{role} Classification Report:")
        print(classification_report(true, pred, zero_division=0))

        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(true, pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{role} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(f'{role.lower()}_confusion_matrix.png')
        plt.close()

        return {
            'Accuracy': accuracy_score(true, pred),
            'Precision': precision_score(true, pred, average='weighted', zero_division=0),
            'Recall': recall_score(true, pred, average='weighted', zero_division=0),
            'F1': f1_score(true, pred, average='weighted', zero_division=0)
        }

    reg_metrics = {
        'Nurse': {
            'MAE': mean_absolute_error(all_true_n, all_pred_n),
            'MSE': mean_squared_error(all_true_n, all_pred_n),
            'R2': r2_score(all_true_n, all_pred_n)
        },
        'Doctor': {
            'MAE': mean_absolute_error(all_true_d, all_pred_d),
            'MSE': mean_squared_error(all_true_d, all_pred_d),
            'R2': r2_score(all_true_d, all_pred_d)
        }
    }

    cls_metrics = {
        'Nurse': generate_report(all_true_n, all_pred_n, 'Nurse'),
        'Doctor': generate_report(all_true_d, all_pred_d, 'Doctor')
    }

    return {'regression': reg_metrics, 'classification': cls_metrics}


def plot_training_progress(metrics_history, eval_epochs):
    plt.figure(figsize=(15, 5))
    metrics = ['MAE', 'MSE', 'R2']
    roles = ['Nurse', 'Doctor']
    for i, metric in enumerate(metrics):
        plt.subplot(1, 3, i + 1)
        for role in roles:
            values = [m[role][metric] for m in metrics_history]
            plt.plot(eval_epochs, values, label=role, marker='o')
        plt.title(f'{metric} over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.close()


def main():
    # 数据加载
    df = pd.read_csv(DATA_PATH)
    X = df[['scenario', 'lambda', 'mu_nurse', 'mu_doctor',
            's_nurse_max', 's_doctor_max', 'Tmax', 'nurse_price', 'doctor_price']]
    y = df[['optimal_nurses', 'optimal_doctors']].values

    # 数据预处理
    preprocessor = ColumnTransformer(
        transformers=[
            ('scenario', OneHotEncoder(), ['scenario']),
            ('num', StandardScaler(), ['lambda', 'mu_nurse', 'mu_doctor', 'Tmax']),
            ('passthrough', 'passthrough', ['s_nurse_max', 's_doctor_max', 'nurse_price', 'doctor_price'])
        ])

    # 修正后的数据分割流程
    indices = np.arange(len(X))
    X_train_idx, X_test_idx, y_train, y_test = train_test_split(
        indices, y, test_size=0.2, random_state=42
    )

    # 获取原始测试集特征
    X_raw_test = X.iloc[X_test_idx]

    # 预处理流程
    preprocessor.fit(X.iloc[X_train_idx])
    X_train = preprocessor.transform(X.iloc[X_train_idx])
    X_test = preprocessor.transform(X.iloc[X_test_idx])

    if not isinstance(X_train, np.ndarray):
        X_train = X_train.toarray()
    if not isinstance(X_test, np.ndarray):
        X_test = X_test.toarray()

    max_n = df['s_nurse_max'].max()
    max_d = df['s_doctor_max'].max()
    n_nurse_classes = int(max_n) + 1
    n_doctor_classes = int(max_d) + 1

    train_loader = DataLoader(HospitalDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(HospitalDataset(X_test, y_test), batch_size=BATCH_SIZE)

    model = HospitalStaffingModel(
        input_dim=X_train.shape[1],
        hidden_layers=HIDDEN_LAYERS,
        n_nurse_classes=n_nurse_classes,
        n_doctor_classes=n_doctor_classes
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion_n = nn.CrossEntropyLoss()
    criterion_d = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

    early_stop_patience = 200
    best_loss = float('inf')
    no_improve_epochs = 0

    train_loss_history = []
    val_loss_history = []
    metrics_history = []
    eval_epochs = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for features, (nurse_t, doctor_t) in train_loader:
            features = features.to(device)
            nurse_t = nurse_t.to(device)
            doctor_t = doctor_t.to(device)

            optimizer.zero_grad()
            nurse_logits, doctor_logits = model(features)
            loss = criterion_n(nurse_logits, nurse_t) + criterion_d(doctor_logits, doctor_t)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        metrics = evaluate_during_training(model, test_loader)
        current_val_loss = (metrics['Nurse']['MSE'] + metrics['Doctor']['MSE']) / 2

        metrics_history.append(metrics)
        eval_epochs.append(epoch)
        train_loss_history.append(total_loss / len(train_loader))
        val_loss_history.append(current_val_loss)

        if current_val_loss < best_loss:
            best_loss = current_val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(torch.load('best_model.pth'))
    final_metrics = final_evaluation(model, test_loader, X_raw_test)
    plot_training_progress(metrics_history, eval_epochs)

    print("\nFinal Evaluation Results:")
    for role in ['Nurse', 'Doctor']:
        print(f"\n=== {role} Performance ===")
        print("Regression Metrics:")
        print(f"MAE: {final_metrics['regression'][role]['MAE']:.2f}")
        print(f"MSE: {final_metrics['regression'][role]['MSE']:.2f}")
        print(f"R²: {final_metrics['regression'][role]['R2']:.2f}")
        print("\nClassification Metrics:")
        print(f"Accuracy: {final_metrics['classification'][role]['Accuracy']:.2f}")
        print(f"Precision: {final_metrics['classification'][role]['Precision']:.2f}")
        print(f"Recall: {final_metrics['classification'][role]['Recall']:.2f}")
        print(f"F1: {final_metrics['classification'][role]['F1']:.2f}")

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.title('Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_loss.png')
    plt.close()


if __name__ == '__main__':
    main()