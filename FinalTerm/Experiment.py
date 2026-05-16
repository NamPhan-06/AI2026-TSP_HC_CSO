import time
from TSP_HillClimbing import HillClimbing
# Lưu ý: Sửa tên import cho đúng với tên File và Class bạn đã đặt
from TSP_CSO import CSO

def run_comparison_experiment(num_cities, seed):
    """
    Chạy thực nghiệm so sánh giữa HC Thuần và CSO Thuần.
    Đánh giá dựa trên Best, Average và Worst cases.
    """
    # ==========================================
    # 1. CHẠY HILL CLIMBING THUẦN
    # ==========================================
    hc = HillClimbing(num_cities, seed=seed)
    hc_route, hc_distance, *rest_hc = hc.solve()

    # ==========================================
    # 2. CHẠY CSO THUẦN (Đã chuyển sang Class cho đồng bộ)
    # ==========================================
    # Không cần định nghĩa tsp_fitness ở đây nữa vì Class đã tự xử lý bên trong
    cso = CSO(num_cities=num_cities, seed=seed)
    cso_route, cso_distance = cso.solve()

    # Lấy các thông số đo lường trực tiếp từ object cso (Cực kỳ gọn)
    cso_time_ms = cso.time_measured_ms
    cso_kb = cso.space_measured_kb

    # ==========================================
    # 3. TÍNH TOÁN VÀ ĐÁNH GIÁ ĐỘ TỐI ƯU
    # ==========================================
    improvement = round(((hc_distance - cso_distance) / hc_distance) * 100, 1)

    # Format Text kết quả báo cáo hiển thị
    report_text = f"========= KẾT QUẢ SO SÁNH THỰC TẾ ========\n"
    report_text += "=" * 42 + "\n"
    report_text += f"{'Tiêu chí':<12} | {'HC Thuần':<12} | {'CSO Thuần':<12}\n"
    report_text += "-" * 45 + "\n"
    report_text += f"{'Q.Đường':<12} | {hc_distance:<12.1f} | {cso_distance:<12.1f}\n"
    report_text += f"{'T.Gian(ms)':<12} | {hc.time_measured_ms:<12.2f} | {cso_time_ms:<12.2f}\n"
    report_text += f"{'D.Lượng(KB)':<12} | {hc.space_measured_kb:<12.2f} | {cso_kb:<12.2f}\n"
    report_text += "=" * 42 + "\n\n"

    report_text += "KẾT LUẬN HIỆU NĂNG:\n"
    if improvement > 0:
        report_text += f"- CSO Thuần tối ưu lộ trình tốt hơn {improvement}%.\n"
    elif improvement < 0:
        report_text += f"- Hill Climbing cho kết quả tốt hơn {-improvement}%.\n"
    else:
        report_text += "- Cả hai thuật toán đều hội tụ về cùng một quãng đường.\n"

    history_text = f"{'Vòng':<5} | {'HC Thuần':<10} | {'CSO Thuần':<10}\n"
    history_text += "-" * 32 + "\n"
    max_len = max(len(hc.convergence), len(cso.convergence))
    for i in range(max_len):
        hc_val = hc.convergence[i] if i < len(hc.convergence) else hc.convergence[-1]
        cso_val = cso.convergence[i] if i < len(cso.convergence) else cso.convergence[-1]
        history_text += f"{i:<5} | {hc_val:<10.1f} | {cso_val:<10.1f}\n"

    return {
        "matrix": hc.matrix,
        "hc_route": hc_route,
        "cso_route": cso_route,
        "hc_convergence": hc.convergence,
        "cso_convergence": cso.convergence,
        "report_text": report_text,
        "history_text": history_text  # Trả về mảng text lịch sử
    }