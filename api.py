from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np

app = FastAPI(
    title="Fitness Kalori Tahmin API",
    description="Egzersiz verilerine göre yakılan kaloriyi tahmin eden servis."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Geliştirme ortamı için tüm portlara izin verir
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Eğitilmiş nesneleri yükleme
model = joblib.load('xgb_model.pkl')
ohe = joblib.load('ohe_encoder.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# Gelen istek gövdesinin şeması (Veri Tipleri ve Doğrulama)
class ExerciseInput(BaseModel):
    Age: int = Field(..., ge=10, le=100, example=28)
    Gender: str = Field(..., example="Male") # Male / Female
    Weight_kg: float = Field(..., alias="Weight (kg)", example=75.5)
    Height_m: float = Field(..., alias="Height (m)", example=1.80)
    Max_BPM: int = Field(..., example=178)
    Avg_BPM: int = Field(..., example=145)
    Resting_BPM: int = Field(..., example=62)
    Session_Duration_hours: float = Field(..., alias="Session_Duration (hours)", example=1.2)
    Workout_Type: str = Field(..., example="HIIT") # Yoga, HIIT, Cardio, Strength
    Fat_Percentage: float = Field(..., example=18.5)
    Water_Intake_liters: float = Field(..., alias="Water_Intake (liters)", example=2.0)
    Workout_Frequency_days_week: int = Field(..., alias="Workout_Frequency (days/week)", example=4)
    Experience_Level: int = Field(..., example=2)
    BMI: float = Field(..., example=23.3)

    class Config:
        populate_by_name = True

@app.post("/predict")
def predict_calories(data: ExerciseInput):
    try:
        # 1. Ham veriyi DataFrame'e dönüştür
        raw_dict = data.model_dump(by_alias=True)
        input_df = pd.DataFrame([raw_dict])
        
        # 2. Kategorik değişkenleri OneHotEncode et
        cat_features = input_df[['Workout_Type', 'Gender']]
        encoded_array = ohe.transform(cat_features)
        
        # EKLENEN KISIM: Eğer sıkıştırılmış (sparse) formattaysa, standart matrise (array) çevir
        if hasattr(encoded_array, 'toarray'):
            encoded_array = encoded_array.toarray()
            
        encoded_cols = ohe.get_feature_names_out(['Workout_Type', 'Gender'])
        encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols)
        
        # 3. Sayısal ve dönüştürülmüş kategorik sütunları birleştir
        numeric_df = input_df.drop(columns=['Gender', 'Workout_Type'])
        final_input = pd.concat([numeric_df, encoded_df], axis=1)
        
        # 4. Sütunların modelin eğitildiği sırayla birebir eşleşmesini sağla
        final_input = final_input[feature_columns]
        
        # 5. Tahmin üret
        prediction = model.predict(final_input)[0]
        
        return {
            "status": "success",
            "predicted_calories_burned": round(float(prediction), 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}