from fastapi import FastAPI, HTTPException #untuk bikin RestAPI 
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field # validasi input
from typing import Optional, Dict, Any, List
import pickle # untuk load model yang udh dilatih
import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging # log process untuk debugging 

# Setup logging -> melacak aktivitas dan error yang terjadi selama API berjalan
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="XGBoost Obesity Classification API",
    description="XGBoost API for Obesity Level Prediction Based on Lifestyle and Daily Habits",
    version="1.0.0",
    docs_url="/docs", # untuk akses dokumentasi Swagger
    redoc_url="/redoc"
)

# Biar API bisa diakses dari web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables untuk model dan target encoder
pipeline_model = None
target_encoder = None
feature_info = None

# File paths -> lokasi model dan juga encoder 
PIPELINE_MODEL_PATH = "best_xgb_model.pkl"
TARGET_ENCODER_PATH = "target_encoder.pkl"
FEATURE_INFO_PATH = "model_feature_info.pkl"

from enum import Enum # untuk membatasi pilihan user 

class GenderEnum(str, Enum):
    male = "Male"
    female = "Female"

class YesNoEnum(str, Enum):
    yes = "yes"
    no = "no"

class CAECEnum(str, Enum):
    no = "no"
    sometimes = "Sometimes"
    frequently = "Frequently"
    always = "Always"

class CALCEnum(str, Enum):
    no = "no"
    sometimes = "Sometimes"
    frequently = "Frequently"

class MTRANSEnum(str, Enum):
    public_transportation = "Public_Transportation"
    automobile = "Automobile"
    motorbike = "Motorbike"
    bike = "Bike"
    walking = "Walking"

class HealthData(BaseModel):
    """Model untuk input data kesehatan"""
    Gender: GenderEnum = Field(..., description="Jenis kelamin (Male/Female)")
    Age: float = Field(..., ge=1, le=120, description="Umur (1-120)")
    Height: float = Field(..., ge=0.5, le=3.0, description="Tinggi badan dalam meter")
    Weight: float = Field(..., ge=10, le=300, description="Berat badan (kg)")
    family_history_with_overweight: YesNoEnum = Field(..., description="Riwayat keluarga obesitas (yes/no)")
    FAVC: YesNoEnum = Field(..., description="Konsumsi makanan berkalori tinggi (yes/no)")
    FCVC: float = Field(..., ge=1, le=3, description="Frekuensi konsumsi sayuran (1-3)")
    NCP: float = Field(..., ge=1, le=4, description="Jumlah makanan utama (1-4)")
    CAEC: CAECEnum = Field(..., description="Konsumsi makanan di antara waktu makan")
    SMOKE: YesNoEnum = Field(..., description="Merokok (yes/no)")
    CH2O: float = Field(..., ge=1, le=3, description="Konsumsi air per hari (1-3)")
    SCC: YesNoEnum = Field(..., description="Monitor kalori (yes/no)")
    FAF: float = Field(..., ge=0, le=3, description="Frekuensi aktivitas fisik (0-3)")
    TUE: float = Field(..., ge=0, le=2, description="Waktu penggunaan teknologi (0-2)")
    CALC: CALCEnum = Field(..., description="Konsumsi alkohol")
    MTRANS: MTRANSEnum = Field(..., description="Transportasi yang digunakan")

class PredictionResponse(BaseModel): # jawaban yang akan diberikan ke user 
    """Model untuk response prediksi"""
    prediction: str # prediksi hasil model
    prediction_encoded: int
    confidence: float # seberapa yakin model
    probabilities: Dict[str, float] # probabilitas setiap kelas 
    bmi: float
    bmi_category: str
    recommendations: List[str] # rekomendasi kesehatan
    risk_level: str # level resiko
    timestamp: str # timestamp hasil prediksi 

class HealthStatus(BaseModel):
    """Model untuk status kesehatan API"""
    status: str
    pipeline_model_loaded: bool
    target_encoder_loaded: bool
    available_endpoints: List[str]
    model_info: Optional[Dict[str, Any]] = None

def load_models(): # otomatis dijalankan saat API nyala 
    """Load XGBoost Pipeline model dan target encoder"""
    global pipeline_model, target_encoder, feature_info
    
    try:
        # Load XGBoost Pipeline model 
        if os.path.exists(PIPELINE_MODEL_PATH):
            with open(PIPELINE_MODEL_PATH, 'rb') as f:
                pipeline_model = pickle.load(f)
            logger.info("XGBoost model loaded successfully")
        else:
            logger.error(f"XGBoost pipeline model file not found: {PIPELINE_MODEL_PATH}")
            
        # Load target encoder
        if os.path.exists(TARGET_ENCODER_PATH):
            with open(TARGET_ENCODER_PATH, 'rb') as f:
                target_encoder = pickle.load(f)
            logger.info("The target encoder has been successfully loaded")
        else:
            logger.error(f"Target encoder file not found: {TARGET_ENCODER_PATH}")
            
    except Exception as e:
        logger.error(f"Error loading XGBoost models: {str(e)}")
        raise e

def calculate_bmi(height: float, weight: float) -> tuple:
    bmi = weight / (height ** 2)
    
    if bmi < 18.5:
        category = "Underweight"
    elif 18.5 <= bmi < 25:
        category = "Normal"
    elif 25 <= bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"
    
    return round(bmi, 2), category

def get_recommendations(prediction: str, bmi_category: str) -> List[str]:
    """Generate rekomendasi berdasarkan prediksi"""
    recommendations = []
    
    base_recommendations = [
        "Consult a doctor or nutritionist",
        "Monitor your daily calorie intake",
        "Increase your intake of fruits and vegetables",
        "Drink at least 8 glasses of water per day"
    ]
    
    if "Obesity" in prediction:
        recommendations.extend([
            "Consult a medical specialist as soon as possible",
            "Consider a medically supervised weight loss program",
            "Avoid foods high in sugar and saturated fats",
            "Engage in moderate-intensity exercise for 150 minutes per week",
            "Monitor your progress regularly using the XGBoost model"
        ])

    elif "Overweight" in prediction:
        recommendations.extend([
            "Start eating smaller portions little by little",
            "Try to move more, at least 30 minutes a day",
            "Replace snacks with fruits or nuts"
        ])
    elif "Normal" in prediction:
        recommendations.extend([
            "Maintain a healthy lifestyle",
            "Exercise regularly at least 3 times a week",
            "Maintain a balanced diet"
        ])
        
    else:  # Insufficient weight
        recommendations.extend([
            "Increase your calorie intake with nutritious foods",
            "Consult a nutritionist for a weight gain program",
            "Use supplements only with your doctor’s advice"
        ])
    
    return base_recommendations + recommendations

def get_risk_level(prediction: str) -> str:
    """Identify the risk level"""
    if "Obesity_Type_III" in prediction:
        return "VERY HIGH"
    elif "Obesity_Type_II" in prediction:
        return "HIGH"
    elif "Obesity_Type_I" in prediction or "Overweight_Level_II" in prediction:
        return "MODERATE"
    elif "Overweight_Level_I" in prediction:
        return "LOW"
    else:
        return "MINIMAL"

def prepare_input_dataframe(data: HealthData) -> pd.DataFrame: #ubah input user menjadidataframe 
    input_dict = data.dict()
    
    df = pd.DataFrame([input_dict])
    
    # make sure urutan kolom sesuai dengan yang dipakai saat training 
    expected_columns = [
        'Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE',  # numeric
        'Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC',  # binary
        'CAEC', 'CALC',  # ordinal
        'MTRANS'  # nominal
    ]
    
    # Reorder columns jika perlu
    df = df.reindex(columns=expected_columns)
    
    logger.info(f"Input DataFrame prepared for XGBoost with shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    return df

@app.on_event("startup") # event startup -> saat API dijalankan, model langsung di load otomatis
async def startup_event():
    """Load XGBoost models saat startup"""
    load_models()

@app.get("/", response_model=HealthStatus) # buat cek apakah APU udah jalan dan model sudah di load 
async def root():
    """Health check endpoint"""
    return HealthStatus(
        status="XGBoost API berjalan dengan baik",
        pipeline_model_loaded=pipeline_model is not None,
        target_encoder_loaded=target_encoder is not None,
        available_endpoints=["/predict", "/health", "/model-info", "/docs", "/redoc"],
        model_info={
            "model_type": "Pipeline (Preprocessor + XGBoostClassifier)",
            "target_classes": list(target_encoder.classes_) if target_encoder else []
        }
    )

@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Detailed health check"""
    return HealthStatus(
        status="healthy" if (pipeline_model is not None and target_encoder is not None) else "unhealthy",
        pipeline_model_loaded=pipeline_model is not None,
        target_encoder_loaded=target_encoder is not None,
        available_endpoints=["/predict", "/health", "/model-info", "/docs", "/redoc"],
        model_info={
            "model_type": "Pipeline (Preprocessor + XGBoostClassifier)",
            "target_classes": list(target_encoder.classes_) if target_encoder else [],
            "pipeline_model_path": PIPELINE_MODEL_PATH,
            "target_encoder_path": TARGET_ENCODER_PATH
        }
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_obesity(data: HealthData):
    """Endpoint untuk prediksi obesitas"""
    try:
        # Check if models are loaded
        if pipeline_model is None or target_encoder is None:
            raise HTTPException(
                status_code=500, 
                detail="The XGBoost pipeline model or target encoder has not been loaded. Please try restarting the application."
            )
        
        # Prepare input DataFrame (raw data)
        input_df = prepare_input_dataframe(data)
        
        # Make prediction (XGBoost Pipeline otomatis handle preprocessing!)
        prediction_encoded = pipeline_model.predict(input_df)[0]
        probabilities = pipeline_model.predict_proba(input_df)[0]
        
        # Convert prediction balik ke label asli
        prediction_original = target_encoder.inverse_transform([prediction_encoded])[0]
        
        # Get confidence (max probability)
        confidence = float(max(probabilities)) # ambil nilai confidence tertinggi 
        
        # Create probability dictionary for every class
        prob_dict = {}
        target_classes = target_encoder.classes_
        for i, class_name in enumerate(target_classes):
            prob_dict[class_name] = float(probabilities[i])
        
        # Calculate BMI
        bmi, bmi_category = calculate_bmi(data.Height, data.Weight)
        
        # Get recommendations and risk level
        recommendations = get_recommendations(prediction_original, bmi_category)
        risk_level = get_risk_level(prediction_original)
        
        return PredictionResponse(
            prediction=prediction_original,
            prediction_encoded=int(prediction_encoded),
            confidence=confidence,
            probabilities=prob_dict,
            bmi=bmi,
            bmi_category=bmi_category,
            recommendations=recommendations,
            risk_level=risk_level,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e: # kalau ada error waktu prediksi 
        logger.error(f"Error in XGBoost prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in XGBoost prediction: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded XGBoost Pipeline model"""
    if pipeline_model is None:
        raise HTTPException(status_code=500, detail="XGBoost pipeline model is not loaded yet")
    
    model_info = {
        "pipeline_steps": [step[0] for step in pipeline_model.steps],
        "model_type": str(type(pipeline_model.named_steps['classifier']).__name__),
        "target_classes": list(target_encoder.classes_) if target_encoder else [],
    }
    
    # Get XGBoost classifier info if available
    classifier = pipeline_model.named_steps.get('classifier')
    if classifier:
        # XGBoost specific parameters
        try:
            if hasattr(classifier, 'feature_importances_'):
                model_info["feature_importances"] = classifier.feature_importances_.tolist()
            if hasattr(classifier, 'n_estimators'):
                model_info["n_estimators"] = classifier.n_estimators
            if hasattr(classifier, 'max_depth'):
                model_info["max_depth"] = classifier.max_depth
            if hasattr(classifier, 'learning_rate'):
                model_info["learning_rate"] = classifier.learning_rate
            if hasattr(classifier, 'subsample'):
                model_info["subsample"] = classifier.subsample
            if hasattr(classifier, 'colsample_bytree'):
                model_info["colsample_bytree"] = classifier.colsample_bytree
            if hasattr(classifier, 'random_state'):
                model_info["random_state"] = classifier.random_state
            if hasattr(classifier, 'objective'):
                model_info["objective"] = classifier.objective
            if hasattr(classifier, 'eval_metric'):
                model_info["eval_metric"] = classifier.eval_metric
        except Exception as e:
            logger.warning(f"⚠️ Could not extract XGBoost parameters: {str(e)}")
    
    # Add feature info if available
    if feature_info:
        model_info["feature_categories"] = {
            "numeric_features": feature_info.get('numeric_features', []),
            "binary_features": feature_info.get('binary_features', []),
            "ordinal_features": feature_info.get('ordinal_features', []),
            "nominal_features": feature_info.get('nominal_features', [])
        }
    
    return model_info

# ketika ada error saat API dijalankan 
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception) # backup handler untuk error 
async def general_exception_handler(request, exc):
    logger.error(f"Unexpected error in XGBoost API: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

# # untuk jalanin aplikasi lewat terminal
if __name__ == "__main__": # ngecek apakah file Python ini dijalankan langsung dari terminal
    import uvicorn # web server yang bisa menjalankan aplikasi FastAPI -> dia yang akan menjalankan server kita
    
    # Run the XGBoost application
    uvicorn.run(
        "main:app", # jalanin aplikasi FastAPI yang ada di file main.py, objeknya bernama app
        host="0.0.0.0",
        port=8000,
        reload=True,  # saat ada perubahan kode, server langsung restart otomatis
        log_level="info"
    )
    





