"""
EDA Visualizations - NYC Taxi Trip Duration
Chạy file này để tạo tất cả các biểu đồ cần thiết cho báo cáo.
Yêu cầu: file data/train.csv phải tồn tại trong thư mục hiện hành.

Cách chạy: python eda_visualizations.py
Kết quả: các file ảnh sẽ được lưu vào thư mục output_images/
"""

import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

# ── Cấu hình chung ──────────────────────────────────────────────────────────
os.makedirs("output_images", exist_ok=True)
sns.set_style("whitegrid")
BLUE = "#1f4e8c"       # màu cột chính
RED  = "#c0392b"       # màu điểm ngoại lệ / đường tham chiếu
FIG_DPI = 150

def save(name):
    plt.tight_layout()
    path = f"output_images/{name}.png"
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  → Đã lưu: {path}")


# ── Đọc & tiền xử lý cơ bản ─────────────────────────────────────────────────
print("Đang đọc dữ liệu...")
df = pd.read_csv("data/train.csv")
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
df["trip_duration_log"] = np.log1p(df["trip_duration"])
df["store_and_fwd_flag"] = df["store_and_fwd_flag"].map({"Y": 1, "N": 0})

# Feature engineering cơ bản
df["hour"]        = df["pickup_datetime"].dt.hour
df["day_of_week"] = df["pickup_datetime"].dt.weekday   # 0=Mon … 6=Sun
df["month"]       = df["pickup_datetime"].dt.month
df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    d = np.sin((lat2-lat1)*0.5)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lng2-lng1)*0.5)**2
    return 2 * 6371 * np.arcsin(np.sqrt(d))

df["dist_haversine"] = haversine_array(
    df["pickup_latitude"], df["pickup_longitude"],
    df["dropoff_latitude"], df["dropoff_longitude"])

# Bỏ outlier rõ ràng để biểu đồ đẹp hơn
df_clean = df[(df["trip_duration"] > 60) & (df["trip_duration"] < 86400)].copy()

print(f"Số dòng gốc : {len(df):,}")
print(f"Sau lọc     : {len(df_clean):,}")
print()

# ════════════════════════════════════════════════════════════════════════════
# 1. MÔ TẢ DỮ LIỆU — bảng missing/unique (heatmap kiểu báo cáo)
# ════════════════════════════════════════════════════════════════════════════
print("1. Vẽ bảng mô tả dữ liệu...")
cols_info = ["id","vendor_id","pickup_datetime","dropoff_datetime","passenger_count",
             "pickup_longitude","pickup_latitude","dropoff_longitude",
             "dropoff_latitude","store_and_fwd_flag","trip_duration"]

info = pd.DataFrame({
    "dtypes"   : df[cols_info].dtypes.astype(str),
    "missing#" : df[cols_info].isnull().sum(),
    "missing%" : (df[cols_info].isnull().mean()*100).round(4),
    "uniques"  : df[cols_info].nunique(),
    "count"    : df[cols_info].count(),
})

fig, ax = plt.subplots(figsize=(12, 4))
ax.axis("off")
tbl = ax.table(
    cellText  = info.values,
    rowLabels = info.index,
    colLabels = info.columns,
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.2, 1.6)

# Tô màu cột "missing#" nếu > 0
for (r, c), cell in tbl._cells.items():
    if r == 0:
        cell.set_facecolor("#1f4e8c"); cell.set_text_props(color="white", fontweight="bold")
    if c == 2 and r > 0:                        # missing% column
        val = float(info.iloc[r-1, 2])
        cell.set_facecolor("#d9534f" if val > 0 else "#dff0d8")
    if c == 4 and r > 0:                        # count column (highlight high)
        cell.set_facecolor("#2196a4"); cell.set_text_props(color="white")

ax.set_title("Mô tả dữ liệu huấn luyện (train.csv)", fontsize=13, fontweight="bold", pad=12)
save("01_data_description")

# ════════════════════════════════════════════════════════════════════════════
# 2. PHÂN PHỐI BIẾN MỤC TIÊU — trip_duration (gốc + log)
# ════════════════════════════════════════════════════════════════════════════
print("2. Phân phối trip_duration...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df_clean["trip_duration"].clip(upper=5000),
             bins=80, color=BLUE, edgecolor="none")
axes[0].set_title("Phân phối trip_duration (giây)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("trip_duration (giây)")
axes[0].set_ylabel("Số lượng")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

axes[1].hist(df_clean["trip_duration_log"], bins=80, color=BLUE, edgecolor="none")
axes[1].set_title("Phân phối log(trip_duration+1)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("log(trip_duration + 1)")
axes[1].set_ylabel("Số lượng")

save("02_trip_duration_distribution")

# ════════════════════════════════════════════════════════════════════════════
# 3. PHÂN PHỐI THEO GIỜ — trip_duration vs hour
# ════════════════════════════════════════════════════════════════════════════
print("3. Scatter: trip_duration vs hour...")
fig, ax = plt.subplots(figsize=(12, 5))
sample = df_clean.sample(min(30_000, len(df_clean)), random_state=42)
ax.scatter(sample["hour"], sample["trip_duration"].clip(upper=5000),
           alpha=0.08, s=8, color=BLUE)
ax.set_xlabel("Giờ trong ngày (0–23)")
ax.set_ylabel("trip_duration (giây)")
ax.set_title("trip_duration theo giờ đón khách", fontsize=12, fontweight="bold")
ax.set_xticks(range(24))
save("03_duration_by_hour")

# ════════════════════════════════════════════════════════════════════════════
# 4. PHÂN PHỐI THEO VENDOR_ID
# ════════════════════════════════════════════════════════════════════════════
print("4. Scatter: trip_duration vs vendor_id...")
fig, ax = plt.subplots(figsize=(8, 5))
sample = df_clean.sample(min(20_000, len(df_clean)), random_state=1)
for vid in sorted(df_clean["vendor_id"].unique()):
    sub = sample[sample["vendor_id"] == vid]
    ax.scatter([vid]*len(sub), sub["trip_duration"].clip(upper=5000),
               alpha=0.08, s=8, label=f"Vendor {vid}", color=BLUE if vid==1 else RED)
ax.set_xlabel("Vendor ID")
ax.set_ylabel("trip_duration (giây)")
ax.set_title("trip_duration theo Vendor ID", fontsize=12, fontweight="bold")
ax.set_xticks([1, 2])
save("04_duration_by_vendor")

# ════════════════════════════════════════════════════════════════════════════
# 5. PHÂN PHỐI THEO PASSENGER_COUNT
# ════════════════════════════════════════════════════════════════════════════
print("5. Scatter: trip_duration vs passenger_count...")
fig, ax = plt.subplots(figsize=(10, 5))
sample = df_clean.sample(min(30_000, len(df_clean)), random_state=2)
for pc in sorted(df_clean["passenger_count"].unique()):
    sub = sample[sample["passenger_count"] == pc]
    ax.scatter([pc]*len(sub), sub["trip_duration"].clip(upper=5000),
               alpha=0.12, s=8, color=BLUE)
ax.set_xlabel("Số hành khách")
ax.set_ylabel("trip_duration (giây)")
ax.set_title("trip_duration theo số lượng hành khách", fontsize=12, fontweight="bold")
save("05_duration_by_passenger_count")

# ════════════════════════════════════════════════════════════════════════════
# 6. PHÂN PHỐI THEO THÁNG
# ════════════════════════════════════════════════════════════════════════════
print("6. Scatter: trip_duration vs month...")
fig, ax = plt.subplots(figsize=(12, 5))
sample = df_clean.sample(min(30_000, len(df_clean)), random_state=3)
ax.scatter(sample["month"], sample["trip_duration"].clip(upper=5000),
           alpha=0.1, s=8, color=BLUE)
ax.set_xlabel("Tháng")
ax.set_ylabel("trip_duration (giây)")
ax.set_title("trip_duration theo tháng trong năm", fontsize=12, fontweight="bold")
ax.set_xticks(range(1, 13))
save("06_duration_by_month")

# ════════════════════════════════════════════════════════════════════════════
# 7. PHÂN PHỐI THEO NGÀY TRONG TUẦN
# ════════════════════════════════════════════════════════════════════════════
print("7. Scatter: trip_duration vs day_of_week...")
fig, ax = plt.subplots(figsize=(12, 5))
sample = df_clean.sample(min(30_000, len(df_clean)), random_state=4)
ax.scatter(sample["day_of_week"], sample["trip_duration"].clip(upper=5000),
           alpha=0.1, s=8, color=BLUE)
ax.set_xlabel("Ngày trong tuần (0=Thứ 2 … 6=Chủ Nhật)")
ax.set_ylabel("trip_duration (giây)")
ax.set_title("trip_duration theo ngày trong tuần", fontsize=12, fontweight="bold")
ax.set_xticks(range(7))
ax.set_xticklabels(["T2","T3","T4","T5","T6","T7","CN"])
save("07_duration_by_dayofweek")

# ════════════════════════════════════════════════════════════════════════════
# 8. PHÂN PHỐI THEO KHOẢNG CÁCH HAVERSINE
# ════════════════════════════════════════════════════════════════════════════
print("8. Scatter: trip_duration vs dist_haversine...")
fig, ax = plt.subplots(figsize=(12, 5))
sample = df_clean.sample(min(40_000, len(df_clean)), random_state=5)
ax.scatter(sample["dist_haversine"].clip(upper=40),
           sample["trip_duration"].clip(upper=5000),
           alpha=0.05, s=6, color=BLUE)
ax.set_xlabel("Khoảng cách Haversine (km)")
ax.set_ylabel("trip_duration (giây)")
ax.set_title("trip_duration theo khoảng cách đường chim bay", fontsize=12, fontweight="bold")
save("08_duration_by_distance")

# ════════════════════════════════════════════════════════════════════════════
# 9. TƯƠNG QUAN (Heatmap)
# ════════════════════════════════════════════════════════════════════════════
print("9. Correlation heatmap...")
corr_cols = ["trip_duration","vendor_id","passenger_count","store_and_fwd_flag",
             "hour","day_of_week","month","is_weekend","dist_haversine","trip_duration_log"]
corr = df_clean[corr_cols].corr()

fig, ax = plt.subplots(figsize=(12, 9))
mask = np.zeros_like(corr, dtype=bool)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, linewidths=0.4, ax=ax,
            annot_kws={"size": 8})
ax.set_title("Ma trận tương quan giữa các đặc trưng", fontsize=13, fontweight="bold")
save("09_correlation_heatmap")

# ════════════════════════════════════════════════════════════════════════════
# 10. LỰA CHỌN ĐẶC TRƯNG — bảng hệ số tương quan với trip_duration
# ════════════════════════════════════════════════════════════════════════════
print("10. Feature correlation table...")
feat_corr = df_clean[corr_cols].corr()["trip_duration"].drop("trip_duration").abs().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 4))
ax.axis("off")
tbl2 = ax.table(
    cellText  = [[f"{v:.4f}"] for v in feat_corr.values],
    rowLabels = list(feat_corr.index),
    colLabels = ["Correlation (absolute)"],
    loc="center",
    cellLoc="center",
)
tbl2.auto_set_font_size(False)
tbl2.set_fontsize(10)
tbl2.scale(1.3, 1.7)

for (r, c), cell in tbl2._cells.items():
    if r == 0:
        cell.set_facecolor("#1f4e8c"); cell.set_text_props(color="white", fontweight="bold")
    elif r > 0 and c == 0:
        val = feat_corr.iloc[r-1]
        if val >= 0.3:
            cell.set_facecolor("#2196a4"); cell.set_text_props(color="white")

ax.set_title("Hệ số tương quan tuyệt đối với trip_duration", fontsize=12, fontweight="bold", pad=10)
save("10_feature_correlation_table")

# ════════════════════════════════════════════════════════════════════════════
# 11. PHÂN PHỐI store_and_fwd_flag
# ════════════════════════════════════════════════════════════════════════════
print("11. trip_duration vs store_and_fwd_flag...")
fig, ax = plt.subplots(figsize=(8, 5))
sample = df_clean.sample(min(20_000, len(df_clean)), random_state=6)
for flag in [0, 1]:
    sub = sample[sample["store_and_fwd_flag"] == flag]
    ax.scatter([flag]*len(sub), sub["trip_duration"].clip(upper=5000),
               alpha=0.1, s=8, color=BLUE if flag==0 else RED)
ax.set_xticks([0, 1])
ax.set_xticklabels(["N (0)", "Y (1)"])
ax.set_xlabel("store_and_fwd_flag")
ax.set_ylabel("trip_duration (giây)")
ax.set_title("trip_duration theo store_and_fwd_flag", fontsize=12, fontweight="bold")
save("11_duration_by_store_flag")

# ════════════════════════════════════════════════════════════════════════════
# 12. SO SÁNH MÔ HÌNH (SỐ LIỆU THỰC TẾ TỪ LOG)
# ════════════════════════════════════════════════════════════════════════════
print("12. Biểu đồ so sánh mô hình (Dữ liệu thực tế)...")

# Dữ liệu trích xuất chính xác từ log bạn gửi
# Thêm XGBoost vào danh sách baseline
models      = ["Linear\nRegression", "LightGBM", "CatBoost", "XGBoost"]
valid_rmse  = [0.589455, 0.365003, 0.364873, 0.357779] 


threshold   = 0.40  # Ngưỡng mục tiêu RMSLE < 0.40

x = np.arange(len(models))
w = 0.35
fig, ax = plt.subplots(figsize=(12, 5)) # Tăng chiều rộng lên 12 để đủ chỗ cho 5 model

bars_v = ax.bar(x + w/2, valid_rmse, w, label="Valid RMSLE", color=BLUE)

# Ghi số liệu cụ thể trên đầu cột
for bar in bars_v:
    h = bar.get_height()
    ax.text(bar.get_x()+bar.get_width()/2, h+0.005, f"{h:.4f}",
            ha="center", va="bottom", fontsize=8, fontweight='bold')

ax.axhline(threshold, color=RED, linewidth=1.4, linestyle="--", label=f"Ngưỡng đạt = {threshold}")
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9)
ax.set_ylabel("RMSLE (Càng thấp càng tốt)")
ax.set_title("So sánh các mô hình Baseline (Kết quả thực tế)", fontsize=12, fontweight="bold")
ax.legend()
ax.set_ylim(0, max(valid_rmse)*1.2)
save("12_model_comparison_baseline")

# ════════════════════════════════════════════════════════════════════════════
# 13. QUÁ TRÌNH TỐI ƯU MÔ HÌNH XGBOOST
# ════════════════════════════════════════════════════════════════════════════
print("13. Biểu đồ quá trình tối ưu XGBoost...")

# Trích xuất từ log các giai đoạn của riêng XGBoost:
stages      = ["XGB Baseline", "Feature Eng.", "Fine-tuning"]
xgb_results = [0.3571, 0.3528, 0.3501]

x2 = np.arange(len(stages))
fig, ax = plt.subplots(figsize=(9, 5))

# Vẽ biểu đồ cột đơn
bars = ax.bar(stages, xgb_results, color=[BLUE, "#2ecc71", "#f1c40f"], width=0.5)

for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x()+bar.get_width()/2, h+0.005, f"{h:.4f}", 
            ha="center", va="bottom", fontweight="bold")

# Đường kẻ kết nối để thấy sự đi xuống của sai số
ax.plot(stages, xgb_results, marker='o', color=RED, linestyle='-', linewidth=2)

ax.set_ylabel("RMSLE")
ax.set_title("Sự cải thiện của XGBoost qua từng bước", fontsize=12, fontweight="bold")
ax.set_ylim(0.34, 0.37) 
save("13_xgboost_tuning_comparison")

print("\nHoàn tất! Tất cả ảnh đã được lưu vào thư mục output_images/")
