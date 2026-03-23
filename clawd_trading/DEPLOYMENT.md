# Deployment Guide for Clawd Trading

## Quick Deploy to Railway (Recommended)

### Option 1: Railway CLI

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Create a new project
railway init

# Set environment variables
railway variables set FIREBASE_SERVICE_ACCOUNT="$(cat firebase_service_account.json)"
railway variables set POLYGON_API_KEY="your_polygon_key"
railway variables set TRADING_MODE="paper"
railway variables set SYMBOLS="NAS100,SPY"

# Deploy
railway up
```

### Option 2: Railway Dashboard

1. Go to https://railway.app
2. Create new project
3. Click "Deploy from GitHub repo"
4. Select your repository
5. Add environment variables in the Variables tab:
   - `FIREBASE_SERVICE_ACCOUNT` - Full JSON content of service account
   - `POLYGON_API_KEY` - Your Polygon.io API key
   - `TRADING_MODE` - Set to `paper` or `live`
   - `SYMBOLS` - Comma-separated symbols (e.g., `NAS100,SPY`)

## Deploy to Heroku

```bash
# Install Heroku CLI and login
heroku login

# Create Heroku app
heroku create clawd-trading-engine

# Set environment variables
heroku config:set FIREBASE_SERVICE_ACCOUNT="$(cat firebase_service_account.json)"
heroku config:set POLYGON_API_KEY="your_polygon_key"
heroku config:set TRADING_MODE="paper"

# Deploy
git push heroku main
```

## Environment Variables Required

| Variable | Required | Description |
|----------|----------|-------------|
| `FIREBASE_SERVICE_ACCOUNT` | Yes | JSON content of Firebase service account key |
| `FIREBASE_RTDB_URL` | No | Firebase Realtime DB URL (default provided) |
| `FIREBASE_PROJECT_ID` | No | Firebase project ID (default provided) |
| `POLYGON_API_KEY` | No | Polygon.io API key (for live data) |
| `TRADELOCKER_API_KEY` | No | TradeLocker API key (for execution) |
| `TRADELOCKER_ACCOUNT_ID` | No | TradeLocker account ID |
| `TRADING_MODE` | No | `paper` or `live` (default: `paper`) |
| `SYMBOLS` | No | Comma-separated symbols (default: `NAS100,SPY`) |

## Getting Firebase Service Account

1. Go to https://console.firebase.google.com
2. Select your project
3. Go to Project Settings > Service Accounts
4. Click "Generate new private key"
5. Save the JSON file
6. Set the entire JSON content as `FIREBASE_SERVICE_ACCOUNT` environment variable

## Verifying Deployment

After deployment, check:

1. **Health endpoint**: `https://your-app-url/`
2. **Status endpoint**: `https://your-app-url/status`
3. **Firebase RTDB**: Data should appear at `https://clawd-trading-7b8de-default-rtdb.firebaseio.com/live_state/`
4. **Frontend**: Dashboard should show "LIVE" status

## Troubleshooting

### Firebase Connection Issues

```bash
# Check logs
railway logs

# Verify environment variables
railway variables
```

### Missing Dependencies

```bash
# Rebuild
railway up --build
```

## Frontend Deployment

The frontend (index.html) is deployed via GitHub Pages:

1. Push changes to GitHub
2. Go to Settings > Pages
3. Source: Deploy from branch (main)
4. Folder: / (root)
5. Wait for deployment

URL: https://ceyre-boop.github.io/quant/
