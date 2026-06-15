# Deployment Guide

This project is split into:
- Frontend: React + Vite in flypredict-ui
- Backend API: Python in app.py

## 1) Deploy Backend (Render)

1. Push this repository to GitHub.
2. In Render, create a new Web Service from your repo.
3. Configure:
   - Root Directory: .
   - Runtime: Python (Python 3.10+ is required, configured automatically via `.python-version`)
   - Build Command: `pip install -r requirements.txt` (Optimized to exclude unused packages like streamlit/plotly to stay within Render's free tier memory limit)
   - Start Command: `python app.py`
4. Add environment variables:
   - HOST=0.0.0.0
   - PORT=10000
5. Deploy and copy your backend URL, for example:
   - https://your-backend-service.onrender.com
6. Test endpoints:
   - /health
   - /metadata

## 2) Deploy Frontend (Vercel)

1. In Vercel, import the same GitHub repository.
2. Configure:
   - Root Directory: flypredict-ui
   - Framework Preset: Vite
   - Build Command: npm run build
   - Output Directory: dist
3. Add environment variable:
   - VITE_API_BASE_URL=https://your-backend-service.onrender.com
4. Deploy.

Note:
- vercel.json is already included for SPA routing support.

## 3) CORS

Backend already sends:
- Access-Control-Allow-Origin: *
So frontend hosted on Vercel can call the API.

## 4) Post-deploy checklist

- Open deployed frontend and login.
- Go to prediction page.
- Submit a prediction and confirm result card appears.
- Verify browser network requests point to your deployed backend URL.

## 5) Local environment for frontend

Use flypredict-ui/.env (or .env.local):

VITE_API_BASE_URL=http://127.0.0.1:8000
