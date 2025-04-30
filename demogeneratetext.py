# ask_ollama.py

import requests

def ask_ollama_http(user_request):
    full_prompt = f"""
Dựa trên các thông tin du lịch dưới đây, hãy trả lời yêu cầu của người dùng:

1. Tour Hà Nội: tổ chức nhiều sự kiện lễ hội như "Hồi sinh" cây xà cừ, triển lãm ''Non sông liền một dải'', lễ hội Cowboy... diễn ra từ tháng 4 đến tháng 5/2025 tại Hà Nội.

2. Tour Phú Quốc: Phục hồi mạnh sau giai đoạn bị du khách quay lưng 2023-2024. Năm 2025, lượng khách quốc tế tăng, dịch vụ được nâng cấp, nhiều chương trình ưu đãi. Tuy nhiên, giá dịch vụ vẫn cao hơn mặt bằng chung do yếu tố đảo.

3. Tour Hồ Tràm: tổ chức sự kiện "V-Fest Evolution" 2-3/5 với các hoạt động như trình diễn khinh khí cầu, âm nhạc, lướt sóng, motor nước, check-in cầu ngắm biển dài nhất châu Á, khu ẩm thực và giải trí tại Hamptons Plaza Hồ Tràm.

4. Tour Hạ Long: du thuyền nghệ thuật Indochine Grand ra mắt với nhiều tiện ích và trải nghiệm du lịch biển cao cấp.

Yêu cầu của người dùng: {user_request}
    """

    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "llama3",  # hoặc model bạn đang dùng tại Ollama server
        "prompt": full_prompt,
        "stream": False
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "Không nhận được phản hồi từ mô hình.")
    except Exception as e:
        return f"Lỗi: {str(e)}"

if __name__ == "__main__":
    # Ví dụ test nhanh
    user_input = input("Enter a question: ")
    print(ask_ollama_http(user_input))
