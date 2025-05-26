from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import joblib
import tensorflow as tf

app = FastAPI()

# Load model dan scaler
model_health = tf.keras.models.load_model('model_asuransi_kesehatan.h5', custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
scaler_health = joblib.load('scaler_asuransi_kesehatan.pkl')

model_vehicle = tf.keras.models.load_model('model_asuransi_kendaraan.h5', custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
scaler_vehicle = joblib.load('scaler_asuransi_kendaraan.pkl')

model_property = tf.keras.models.load_model('model_asuransi_properti.h5', custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
scaler_property = joblib.load('scaler_properti.pkl')

# Input Schema
class InputKesehatan(BaseModel):
    umur: int = Field(..., ge=0, le=120)
    penghasilan: int = Field(..., ge=0)
    merokok: str
    bmi: float = Field(..., ge=10.0, le=60.0)
    penyakit_kronis: str

class InputKendaraan(BaseModel):
    umur_kendaraan: int = Field(..., ge=0, le=50)
    harga_kendaraan: int = Field(..., ge=0)
    pengemudi_muda: str
    area_risiko: str

class InputProperti(BaseModel):
    umur_bangunan: int = Field(..., ge=0, le=100)
    luas_bangunan: int = Field(..., ge=1)
    tipe_material: str
    lokasi_risiko: str
    nilai_properti: int = Field(..., ge=0)

# Mapping Functions
def map_merokok(val): return {"tidak": 0, "ya": 1}.get(val.lower(), 0)
def map_penyakit_kronis(val): return {"tidak": 0, "ya": 1}.get(val.lower(), 0)
def map_tipe_material(val): return {"standar": 0, "tahan api": 1}.get(val.lower(), 0)
def map_lokasi_risiko_properti(val): return {"aman": 0, "rawan": 1}.get(val.lower(), 0)
def map_pengemudi_muda(val): return {"di bawah 25": 1, "di atas 25": 0}.get(val.lower(), 0)
def map_area_risiko(val): return {"aman": 0, "sedang": 1, "rawan": 2}.get(val.lower(), 0)

# Endpoint Asuransi Kesehatan
@app.post("/predict/kesehatan")
def predict_kesehatan(data: InputKesehatan):
    try:
        input_arr = np.array([[data.umur, data.penghasilan, map_merokok(data.merokok),
                               data.bmi, map_penyakit_kronis(data.penyakit_kronis)]])
        scaled = scaler_health.transform(input_arr)
        pred = model_health.predict(scaled)

        if len(pred[0]) < 4:
            raise HTTPException(status_code=500, detail="Output model tidak sesuai (kurang dari 4 elemen).")

        kelas_risiko = int(np.argmax(pred[0][:3]))
        premi = float(pred[0][3])

        return {
            "kelas_risiko": {0: "rendah", 1: "sedang", 2: "tinggi"}[kelas_risiko],
            "estimated_premi": round(premi)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint Asuransi Kendaraan
@app.post("/predict/kendaraan")
def predict_kendaraan(data: InputKendaraan):
    try:
        input_arr = np.array([[data.umur_kendaraan, data.harga_kendaraan,
                               map_pengemudi_muda(data.pengemudi_muda),
                               map_area_risiko(data.area_risiko)]])
        scaled = scaler_vehicle.transform(input_arr)
        pred = model_vehicle.predict(scaled)

        if len(pred[0]) < 4:
            raise HTTPException(status_code=500, detail="Output model tidak sesuai.")

        kelas_risiko = int(np.argmax(pred[0][:3]))
        premi = float(pred[0][3])

        return {
            "kelas_risiko": {0: "rendah", 1: "sedang", 2: "tinggi"}[kelas_risiko],
            "estimated_premi": round(premi)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint Asuransi Properti
@app.post("/predict/properti")
def predict_properti(data: InputProperti):
    try:
        input_arr = np.array([[data.umur_bangunan, data.luas_bangunan,
                               map_tipe_material(data.tipe_material),
                               map_lokasi_risiko_properti(data.lokasi_risiko),
                               data.nilai_properti]])
        scaled = scaler_property.transform(input_arr)
        pred = model_property.predict(scaled)

        if len(pred[0]) < 4:
            raise HTTPException(status_code=500, detail="Output model tidak sesuai.")

        kelas_risiko = int(np.argmax(pred[0][:3]))
        premi = float(pred[0][3])

        return {
            "kelas_risiko": {0: "rendah", 1: "sedang", 2: "tinggi"}[kelas_risiko],
            "estimated_premi": round(premi)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
