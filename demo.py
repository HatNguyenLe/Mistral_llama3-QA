# from sentence_transformers import SentenceTransformer, util
# import torch

# # Load model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Danh sách tour
# tours = [
#     "Tour Đà Lạt 3 ngày 2 đêm, khám phá thung lũng tình yêu",
#     "Tour Hạ Long 2 ngày 1 đêm, du thuyền trên Vịnh",
#     "Tour Hội An, khám phá phố cổ và biển Cửa Đại",
#     "Tour Phú Quốc nghỉ dưỡng 4 ngày 3 đêm",
#     "Tour Sa Pa leo Fansipan 3 ngày"
# ]

# tour_embeddings = model.encode(tours, convert_to_tensor=True)


# user_input = input("Enter a question: ")
# query_embedding = model.encode(user_input, convert_to_tensor=True)

# cosine_scores = util.cos_sim(query_embedding, tour_embeddings)

# top_results = torch.topk(cosine_scores, k=3)

# print("\nCác tour phù hợp nhất:")
# for score, idx in zip(top_results.values[0], top_results.indices[0]):
#     print(f"- {tours[idx]} (Độ tương đồng: {score.item():.4f})")
# --------------- 
from sentence_transformers import SentenceTransformer, util
import torch
import subprocess

# Hàm gọi Ollama để yêu cầu phân tích
def ask_ollama(prompt, model='mistral'):
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return result.stdout.decode("utf-8")

# Hàm so sánh các tour bằng Ollama
def compare_tours_with_ollama(tours, user_request):
    tours_text = "\n".join([f"- Tour {i+1}: {tour}" for i, tour in enumerate(tours)])
    full_prompt = f"""
Tôi muốn đi du lịch với yêu cầu: "{user_request}"

Dưới đây là các tour đề xuất:
{tours_text}

Hãy giúp tôi phân tích mỗi tour:
+ Ưu điểm
+ Nhược điểm

Trình bày ngắn gọn, rõ ràng, dễ so sánh.
    """
    return ask_ollama(full_prompt)

def main():
    # Load model ngôn ngữ
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Danh sách tour mẫu
    tours = [
        "Tour Đà Lạt 3 ngày 2 đêm, khám phá thung lũng tình yêu",
        "Tour Hạ Long 2 ngày 1 đêm, du thuyền trên Vịnh",
        "Tour Hội An, khám phá phố cổ và biển Cửa Đại",
        "Tour Phú Quốc nghỉ dưỡng 4 ngày 3 đêm",
        "Tour Sa Pa leo Fansipan 3 ngày"
    ]

    # Encode tour
    tour_embeddings = model.encode(tours, convert_to_tensor=True)

    # Nhập yêu cầu người dùng
    user_input = input("Enter a question: ")
    query_embedding = model.encode(user_input, convert_to_tensor=True)

    # Tính độ tương đồng cosine
    cosine_scores = util.cos_sim(query_embedding, tour_embeddings)

    # top 3
    top_results = torch.topk(cosine_scores, k=3)

    selected_tours = [tours[idx.item()] for idx in top_results.indices[0]]

    print("\nThe most suitable tours:")
    for score, idx in zip(top_results.values[0], top_results.indices[0]):
        print(f"- {tours[idx.item()]} (Độ tương đồng: {score.item():.4f})")

    # Compare with Ollama + Mistral
    print("\nWaiting a minutes...\n")
    comparison = compare_tours_with_ollama(selected_tours, user_input)
    print(comparison)

if __name__ == "__main__":
    main()
