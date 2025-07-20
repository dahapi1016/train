# === Publication-grade styling ===
import matplotlib as mpl
from functools import partial

import matplotlib.pyplot as plt  # Ensure plt available early
import seaborn as sns

# Use serif fonts common in papers
mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'axes.linewidth': 0.8,
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

# Seaborn context for paper
sns.set_style("ticks", {'grid.linestyle': '--', 'grid.alpha': 0.4})
sns.set_context("paper", font_scale=1.3)

# Color-blind friendly palette
colors = sns.color_palette("colorblind", 3)

_old_savefig = plt.savefig

def _save_show(path, *args, **kwargs):
    """Save figure, show in PyCharm SciView, then close; enforce 600-dpi tight bbox."""
    kwargs.pop("dpi", None)
    kwargs.pop("bbox_inches", None)
    _old_savefig(path, dpi=600, bbox_inches="tight", *args, **kwargs)
    try:
        plt.show(block=False)  # SciView panel in PyCharm
    except TypeError:
        plt.show()
    plt.close()

# Override
plt.savefig = _save_show

# Helper to clean axis aesthetics
def _style_axis(ax):
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Apply helper in key plots (example modification for radar and training stages)
# Modify radar plot usage

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------
# Fake data definition
# ----------------------------------
methods = ["PAN+DNN Hybrid", "Traditional PAN", "Traditional DNN"]
colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

# 1. Radar chart data (normalized performance metrics)
metrics = ["MAE", "MSE", "R2", "Accuracy"]
# Each method: [MAE_inv, MSE_inv, R2, Accuracy]
radar_data = {
    "PAN+DNN Hybrid": [0.85, 0.83, 0.90, 0.88],
    "Traditional PAN": [0.72, 0.70, 0.75, 0.73],
    "Traditional DNN": [0.78, 0.75, 0.80, 0.79]
}

# 2. Line chart data (training and validation loss over epochs)
epochs = np.arange(1, 41)
train_loss = {
    "PAN+DNN Hybrid": 1.5 * np.exp(-epochs/12) + 0.1 * np.random.normal(0, 0.05, len(epochs)),
    "Traditional PAN": 2.2 * np.exp(-epochs/15) + 0.15 * np.random.normal(0, 0.05, len(epochs)),
    "Traditional DNN": 1.9 * np.exp(-epochs/14) + 0.12 * np.random.normal(0, 0.05, len(epochs))
}
val_loss = {m: v + 0.1 for m, v in train_loss.items()}

# 3. Bar chart data (penalty loss)
penalty_loss = {
    "PAN+DNN Hybrid": 4.2,
    "Traditional PAN": 7.8,
    "Traditional DNN": 6.3
}

# ----------------------------------
#  Plot 1: Radar chart
# ----------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw=dict(projection='polar'), figsize=(14, 6))
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

for idx, (method, data) in enumerate(radar_data.items()):
    values = data + data[:1]
    ax = ax1 if idx == 0 else ax2  # split for nurse/doctor? here just duplicate for demonstration
    ax.plot(angles, values, label=method, color=colors[idx], linewidth=2)
    ax.fill(angles, values, color=colors[idx], alpha=0.25)

ax1.set_xticks(angles[:-1])
ax1.set_xticklabels(metrics)
ax1.set_title("Performance Comparison (Nurse)")
ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3)

ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(metrics)
ax2.set_title("Performance Comparison (Doctor)")

plt.tight_layout()
plt.savefig("fake_performance_radar.png", dpi=300)
plt.close()

# ----------------------------------
# Plot 2: Training loss curves
# ----------------------------------
plt.figure(figsize=(10, 6))
for idx, method in enumerate(methods):
    plt.plot(epochs, train_loss[method], label=f"{method} - Train", color=colors[idx], linestyle='-')
    plt.plot(epochs, val_loss[method], label=f"{method} - Val", color=colors[idx], linestyle='--')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curves (Fake Data)")
plt.legend()
plt.tight_layout()
plt.savefig("fake_training_loss_curves.png", dpi=300)
plt.close()

# ----------------------------------
# Plot 3: Penalty loss bar chart
# ----------------------------------
plt.figure(figsize=(8, 6))
loss_values = [penalty_loss[m] for m in methods]
plt.bar(methods, loss_values, color=colors, alpha=0.8)
plt.ylabel("Penalty Loss")
plt.title("Penalty Loss Comparison (Fake Data)")
for i, v in enumerate(loss_values):
    plt.text(i, v + 0.1, f"{v:.1f}", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig("fake_penalty_loss_bar.png", dpi=300)
plt.close()

# ----------------------------------
# Additional fake performance data
# ----------------------------------
# Accuracy (Nurse, Doctor)
accuracy = {
    "nurse": [0.88, 0.74, 0.80],
    "doctor": [0.86, 0.72, 0.78]
}

# Error distributions (prediction - true) for boxplots
np.random.seed(42)
error_dist = {
    "nurse": {
        "PAN+DNN Hybrid": np.random.normal(0, 0.8, 300),
        "Traditional PAN": np.random.normal(0, 1.5, 300),
        "Traditional DNN": np.random.normal(0, 1.2, 300)
    },
    "doctor": {
        "PAN+DNN Hybrid": np.random.normal(0, 0.7, 300),
        "Traditional PAN": np.random.normal(0, 1.4, 300),
        "Traditional DNN": np.random.normal(0, 1.1, 300)
    }
}

# Simulated true vs predicted counts for scatter
true_counts = np.random.randint(0, 11, 150)
scatter_pred = {
    "PAN+DNN Hybrid": true_counts + np.random.normal(0, 0.5, 150),
    "Traditional PAN": true_counts + np.random.normal(0, 1.0, 150),
    "Traditional DNN": true_counts + np.random.normal(0, 0.8, 150)
}

# Complexity data
params = [156000, 45000, 89000]
inference_time = [15.2, 8.5, 12.1]

# ----------------------------------
# Plot 4: Accuracy bar charts
# ----------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.bar(methods, accuracy["nurse"], color=colors, alpha=0.8)
ax1.set_ylim(0, 1)
ax1.set_title("Nurse Prediction Accuracy")
for i, v in enumerate(accuracy["nurse"]):
    ax1.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')

ax2.bar(methods, accuracy["doctor"], color=colors, alpha=0.8)
ax2.set_ylim(0, 1)
ax2.set_title("Doctor Prediction Accuracy")
for i, v in enumerate(accuracy["doctor"]):
    ax2.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig("fake_accuracy_bar.png", dpi=300)
plt.close()

# ----------------------------------
# Plot 5: Error distribution boxplots
# ----------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.boxplot([error_dist["nurse"][m] for m in methods], labels=methods)
ax1.set_title("Nurse Prediction Error Distribution")
ax1.set_ylabel("Prediction Error")
ax1.axhline(0, linestyle='--', color='red', alpha=0.6)

ax2.boxplot([error_dist["doctor"][m] for m in methods], labels=methods)
ax2.set_title("Doctor Prediction Error Distribution")
ax2.set_ylabel("Prediction Error")
ax2.axhline(0, linestyle='--', color='red', alpha=0.6)
plt.tight_layout()
plt.savefig("fake_error_distribution.png", dpi=300)
plt.close()

# ----------------------------------
# Plot 6: Prediction scatter (Nurse)
# ----------------------------------
plt.figure(figsize=(14, 5))
for idx, method in enumerate(methods):
    plt.scatter(true_counts, scatter_pred[method], label=method, alpha=0.6, s=40, color=colors[idx])
plt.plot([0, 10], [0, 10], 'k--', alpha=0.7)
plt.xlabel("True Count")
plt.ylabel("Predicted Count")
plt.title("Nurse Count Prediction Scatter (Fake Data)")
plt.legend()
plt.tight_layout()
plt.savefig("fake_prediction_scatter_nurse.png", dpi=300)
plt.close()

# ----------------------------------
# Plot 7: Model complexity comparison
# ----------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.bar(methods, params, color=colors, alpha=0.8)
ax1.set_ylabel("Number of Parameters")
ax1.set_title("Model Size Comparison")
for i, v in enumerate(params):
    ax1.text(i, v + 2000, f"{v:,}", ha='center', fontweight='bold')

ax2.bar(methods, inference_time, color=colors, alpha=0.8)
ax2.set_ylabel("Inference Time (ms)")
ax2.set_title("Inference Time Comparison")
for i, v in enumerate(inference_time):
    ax2.text(i, v + 0.3, f"{v:.1f} ms", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig("fake_complexity_comparison.png", dpi=300)
plt.close()

# ==================================================
# Extra full-set comparison functions (moved from fake_visualization_full)
# ==================================================

def create_full_radar():
    metrics = ["MAE", "MSE", "R2", "Accuracy"]
    values = {
        "PAN+DNN Hybrid": [0.88, 0.84, 0.92, 0.89],
        "Traditional PAN": [0.70, 0.68, 0.75, 0.72],
        "Traditional DNN": [0.78, 0.75, 0.81, 0.79]
    }
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist(); angles += angles[:1]
    fig, (ax1, ax2) = plt.subplots(1,2,subplot_kw=dict(projection='polar'),figsize=(14,6))
    for idx, m in enumerate(methods):
        for ax,title in ((ax1,"Nurse"),(ax2,"Doctor")):
            v = values[m] + values[m][:1]
            ax.plot(angles, v, color=colors[idx], linewidth=2)
            ax.fill(angles, v, alpha=0.25, color=colors[idx])
            ax.set_xticks(angles[:-1]); ax.set_xticklabels(metrics)
            ax.set_title(f"Performance Radar ({title})")
    ax1.legend(methods, loc='lower center', bbox_to_anchor=(0.5,-0.15), ncol=3, frameon=False)
    _style_axis(ax1); _style_axis(ax2)
    plt.tight_layout(); plt.savefig("fake_full_radar.png", dpi=300); plt.close()


def create_full_accuracy():
    nurse_acc=[0.89,0.74,0.80]; doctor_acc=[0.87,0.72,0.78]
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,6))
    for ax,acc,title in ((ax1,nurse_acc,"Nurse"),(ax2,doctor_acc,"Doctor")):
        ax.bar(methods, acc, color=colors, alpha=0.85)
        ax.set_ylim(0,1); ax.set_title(f"{title} Prediction Accuracy")
        for i,v in enumerate(acc): ax.text(i,v+0.02,f"{v:.2f}",ha='center',fontweight='bold')
    _style_axis(ax1); _style_axis(ax2)
    plt.tight_layout(); plt.savefig("fake_full_accuracy.png"); plt.close()


def create_full_error_box():
    nurse_err=[np.random.normal(0,s,300) for s in (0.8,1.5,1.2)]
    doctor_err=[np.random.normal(0,s,300) for s in (0.7,1.4,1.1)]
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,6))
    ax1.boxplot(nurse_err,labels=methods); ax1.set_title("Nurse Prediction Error"); ax1.axhline(0,ls='--',c='red',alpha=0.6)
    ax2.boxplot(doctor_err,labels=methods); ax2.set_title("Doctor Prediction Error"); ax2.axhline(0,ls='--',c='red',alpha=0.6)
    _style_axis(ax1); _style_axis(ax2)
    plt.tight_layout(); plt.savefig("fake_full_error_box.png"); plt.close()


def create_full_training_loss():
    epochs=np.arange(1,61)
    curves={m:(s*np.exp(-epochs/20)+0.05*np.random.normal(0,0.1,len(epochs))) for m,s in zip(methods,(2.0,2.8,2.4))}
    plt.figure(figsize=(10,6))
    markers=['o','s','^']
    for idx,m in enumerate(methods): plt.plot(epochs,curves[m],color=colors[idx],label=m,marker=markers[idx],markevery=6)
    plt.xlabel("Epoch"); plt.ylabel("Training Loss"); plt.title("Training Loss Comparison (Fake Data)"); plt.legend(); plt.tight_layout(); plt.savefig("fake_full_training_loss.png", dpi=300); plt.close()
    _style_axis(plt.gca())


def create_full_complexity():
    params=[156000,45000,89000]; inf=[15.2,8.5,12.1]
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,6))
    ax1.bar(methods,params,color=colors,alpha=0.85); ax1.set_title("Model Size (Params)")
    for i,v in enumerate(params): ax1.text(i,v+2000,f"{v:,}",ha='center')
    ax2.bar(methods,inf,color=colors,alpha=0.85); ax2.set_title("Inference Time (ms)")
    for i,v in enumerate(inf): ax2.text(i,v+0.3,f"{v:.1f} ms",ha='center')
    _style_axis(ax1); _style_axis(ax2)
    plt.tight_layout(); plt.savefig("fake_full_complexity.png"); plt.close()

# ----------------------------------
# Training stages comparison (four stages)
# ----------------------------------

def create_training_stage_comparison():
    stage1_epochs=np.arange(1,26)
    stage2_epochs=np.arange(26,56)
    stage3_epochs=np.arange(56,86)
    stage4_epochs=np.arange(86,111)
    stage1_loss=3.0*np.exp(-(stage1_epochs-1)/8)+1.2+0.05*np.random.normal(0,0.1,len(stage1_epochs))
    stage2_loss=1.2+0.8*np.exp(-(stage2_epochs-26)/10)+0.04*np.random.normal(0,0.1,len(stage2_epochs))
    stage3_loss=0.4+0.3*np.exp(-(stage3_epochs-56)/12)+0.03*np.random.normal(0,0.1,len(stage3_epochs))
    stage4_loss=0.1+0.2*np.exp(-(stage4_epochs-86)/15)+0.02*np.random.normal(0,0.1,len(stage4_epochs))
    all_epochs=np.concatenate([stage1_epochs,stage2_epochs,stage3_epochs,stage4_epochs])
    all_losses=np.concatenate([stage1_loss,stage2_loss,stage3_loss,stage4_loss])
    plt.figure(figsize=(12,6))
    plt.plot(all_epochs, all_losses, color='#FF6B6B', linewidth=2)
    plt.axvline(25, color='gray', ls='--'); plt.axvline(55,color='gray',ls='--'); plt.axvline(85,color='gray',ls='--')
    plt.title("Four-Stage Training Loss (Fake Data)")
    _style_axis(plt.gca())
    plt.xlabel("Epoch"); plt.ylabel("Training Loss")
    plt.tight_layout(); plt.savefig("fake_training_stages.png"); plt.close()

# ----------------------------------
# master function call update
# ----------------------------------
if __name__=="__main__":
    # existing simple charts
    # (already executed earlier when script imported) – Ensure they run once.
    create_full_radar(); create_full_accuracy(); create_full_error_box(); create_full_training_loss(); create_full_complexity(); create_training_stage_comparison()
    print("All fake charts (extended set) generated.") 