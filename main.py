from flask import Flask, request, jsonify
from pymongo import MongoClient
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from flask_cors import CORS

# Kết nối MongoDB
MONGODB_URI = 'mongodb+srv://khanhduc2902:khanhduc2902@cluster0.03at6jt.mongodb.net/cuahangdientu'
client = MongoClient(MONGODB_URI)
db = client['cuahangdientu']
product_collection = db['products']

# Flask app
app = Flask(__name__)
CORS(app)

def get_user_views_from_mongodb():
    """Trích xuất dữ liệu từ MongoDB và chuẩn hóa."""
    products = list(product_collection.find({}, {'_id': 1, 'userViews': 1}))
    df = pd.DataFrame(products)

    # Xử lý nếu không có dữ liệu hoặc thiếu cột `userViews`
    if df.empty or 'userViews' not in df.columns:
        return pd.DataFrame(columns=['user_id', 'id_product', 'view_count', 'updated_at'])

    # Mở rộng `userViews`
    df = df[df['userViews'].notnull()]  # Loại bỏ sản phẩm không có `userViews`
    user_views = df.explode('userViews')
    user_views_normalized = pd.json_normalize(user_views['userViews'])
    user_views = user_views.reset_index(drop=True)
    user_views_normalized = user_views_normalized.reset_index(drop=True)

    # Chuẩn hóa dữ liệu
    user_views_normalized['id_product'] = user_views['_id']
    user_views_normalized.rename(columns={'uid': 'user_id', 'viewCount': 'view_count', 'updatedAt': 'updated_at'}, inplace=True)
    user_views_normalized['view_count'] = user_views_normalized['view_count'].fillna(0).astype(int)
    user_views_normalized['updated_at'] = pd.to_datetime(user_views_normalized['updated_at'], errors='coerce')
    
    return user_views_normalized[['user_id', 'id_product', 'view_count', 'updated_at']]



def preprocess_data(df):
    """Chuẩn hóa dữ liệu và tạo ma trận người dùng - sản phẩm."""
    current_time = pd.Timestamp.now()

    # Chuẩn hóa dữ liệu
    df['days_since'] = (current_time - df['updated_at']).dt.days.fillna(999)
    df['time_weight'] = np.exp(-df['days_since'] / 90)
    df['weighted_score'] = df['view_count'] * df['time_weight']

    # Loại bỏ các hàng trùng lặp (nếu có)
    # df = df.drop_duplicates(subset=['user_id', 'id_product'], keep='last')
    if df.empty:
     raise ValueError("No valid data found for creating user-product matrix.")

    # Xử lý khi pivot
    user_product_matrix = df.pivot_table(
        index='user_id',       # Chỉ mục: người dùng
        columns='id_product',  # Cột: sản phẩm
        values='weighted_score',  # Giá trị: điểm trọng số
        aggfunc='sum'          # Tổng hợp nếu trùng
    ).fillna(0)
    
    return user_product_matrix

@app.route('/')
def home():
    return "Flask is working!"

@app.route('/api/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        user_id = data.get('user_id', '').strip()

        # Lấy dữ liệu từ MongoDB
        df = get_user_views_from_mongodb()
        if df.empty:
            return jsonify({"error": "No data available"}), 404

        # Chuẩn bị dữ liệu
        user_product_matrix = preprocess_data(df)

        # Kiểm tra sự tồn tại của user_id
        print(user_product_matrix.index.to_list())
        string_ids = [str(obj_id) for obj_id in user_product_matrix.index.to_list()]
        print(string_ids)

        if user_id not in string_ids:
        # Gợi ý sản phẩm phổ biến nhất
            popular_products = df.groupby('id_product')['view_count'].sum().nlargest(5).index
            recommendations = [{"product_id": str(prod), "score": 0} for prod in popular_products]
            return jsonify(recommendations)
        
        # Chuẩn hóa dữ liệu cho KNN
        scaler = MinMaxScaler()
        user_product_matrix_normalized = scaler.fit_transform(user_product_matrix)

        # Tạo mô hình KNN
        model = NearestNeighbors(metric='cosine', algorithm='brute')
        model.fit(user_product_matrix_normalized)
        print("===========2")

        # Lấy sản phẩm gợi ý
        user_index = string_ids.index(user_id)
        distances, indices = model.kneighbors([user_product_matrix_normalized[user_index]], n_neighbors=5)

        # Loại bỏ chính người dùng
        recommended_indices = indices[0][1:]
        recommended_products = user_product_matrix.columns[recommended_indices]

        # Tạo danh sách gợi ý và sắp xếp theo `score` giảm dần
        recommendations = [
            {"product_id": str(prod_id), "score": user_product_matrix.iloc[user_index][prod_id]} 
            for prod_id in recommended_products
        ]

        # Sắp xếp danh sách theo `score` giảm dần
        sorted_recommendations = sorted(recommendations, key=lambda x: x['score'], reverse=True)

        # Trả về kết quả
        return jsonify(sorted_recommendations)


    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
