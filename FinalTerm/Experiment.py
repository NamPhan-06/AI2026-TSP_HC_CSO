import time
from TSP_HillClimbing import HillClimbing
from Hybrid_CSO_HillClimbing import hybrid_algorithm_advanced


def run_comparison_experiment(num_cities, seed):
    """
    Chạy thực nghiệm so sánh giữa HC Thuần và Hybrid CSO.
    Trả về Dictionary chứa các metric và chuỗi text báo cáo.
    """
    # 1. Chạy HC Thuần
    hc = HillClimbing(num_cities, seed=seed)
    hc_route, hc_distance = hc.solve()

    # 2. Chạy Hybrid CSO (Dùng chung hc.matrix để đảm bảo công bằng)
    start_time = time.perf_counter()

    # --- ĐÃ CẬP NHẬT: Hứng thêm biến hybrid_complexity ---
    hybrid_route, hybrid_distance, conv, hybrid_complexity = hybrid_algorithm_advanced(
        hc.matrix, N=30, max_iter=100, hc_rate=0.3
    )
    # -----------------------------------------------------

    end_time = time.perf_counter()
    hybrid_time_ms = round((end_time - start_time) * 1000.0, 2)

    # Lấy dung lượng KB của cả 2 thuật toán
    hc_kb = hc.space_measured_kb
    hybrid_kb = hybrid_complexity["space_measured_kb"]

    # 3. Tính toán % cải thiện quãng đường
    improvement = round(((hc_distance - hybrid_distance) / hc_distance) * 100, 1)

    # 4. Format Text kết quả (Bổ sung thêm cột D.Lượng KB)
    report_text = "KẾT QUẢ SO SÁNH THỰC TẾ\n"
    report_text += "=" * 56 + "\n"
    report_text += f"{'Thuật toán':<12} | {'Q.Đường':<8} | {'T.Gian(ms)':<10} | {'D.Lượng(KB)':<12}\n"
    report_text += "-" * 56 + "\n"
    report_text += f"{'HC Thuần':<12} | {hc_distance:<8.1f} | {hc.time_measured_ms:<10.2f} | {hc_kb:<12.2f}\n"
    report_text += f"{'Hybrid CSO':<12} | {hybrid_distance:<8.1f} | {hybrid_time_ms:<10.2f} | {hybrid_kb:<12.2f}\n"
    report_text += "=" * 56 + "\n\n"

    # 5. Đưa ra nhận xét tự động
    report_text += f"KẾT LUẬN:\n"
    if improvement > 0:
        report_text += f"- Hybrid CSO tối ưu lộ trình tốt hơn {improvement}%.\n"
        report_text += f"- Đánh đổi: Tốn nhiều thời gian và bộ nhớ hơn do \n  duy trì quần thể 30 cá thể.\n"
    elif improvement < 0:
        report_text += f"- Hill Climbing cho kết quả tốt hơn {-improvement}%.\n"
        report_text += f"- (Ghi chú: Thường xảy ra ở bài toán kích thước nhỏ).\n"
    else:
        report_text += f"- Cả hai thuật toán đều hội tụ về cùng một quãng đường.\n"

    return {
        "matrix": hc.matrix,
        "hc_route": hc_route,
        "hybrid_route": hybrid_route,
        "report_text": report_text
    }